"""Declarative release manifests (PR 7).

Release/injection metadata that used to live as hardcoded Python dicts in
``gwcat.fetch`` (Zenodo record IDs, file-name filters, per-release quirks) now
lives in YAML files bundled as package data:

    gwcat/manifests/releases/*.yaml     one per PE data release
    gwcat/manifests/injections/*.yaml   one per injection/selection product

This module loads, validates, and exposes those manifests. ``gwcat.fetch``
builds its release registry from them at import time (see ``fetch._build_registry``);
it contains no per-release data of its own any more.

Adding a new release requires only a new YAML file here -- no changes to
``gwcat.fetch`` or any other downloader code (see ``get_manifest`` below,
which also accepts an arbitrary filesystem path for user-supplied manifests).

Manifest schema
---------------
::

    release: GWTC-5                  # str, required; registry key
    aliases: [GWTC-4]                # list[str], optional; extra registry keys
                                      # that resolve to this same manifest
    observing_runs: [O4b]            # list[str], required, non-empty
    observing_run_label: "O1-O4b"    # str, optional; legacy display string for
                                      # `ReleaseManifest.observing_run`. Defaults
                                      # to "+".join(observing_runs).
    description: "..."               # str, required

    records:                         # list, required, non-empty
      - provider: zenodo             # str, required
        record_id: 20348005          # int, required
        concept_id: 20276105         # int or null, optional (default null)

    products:                        # dict, required, non-empty
      pe_samples:                    # product name -> file-selection spec
        file_patterns: ["*.hdf5"]    # list[str], required, non-empty;
                                      # fnmatch-style globs, OR-matched
        exclude_patterns: []         # list[str], optional; globs, OR-rejected
        contains_all: []             # list[str], optional; substrings that
                                      # must ALL be present (AND)
        suffix: null                 # str or null, optional; exact endswith()
        prefix: null                 # str or null, optional; exact startswith()
        require_event_name: false    # bool, optional; require a GW event id
                                      # (regex GW\\d{6}) somewhere in the name
        exclude_junk: false          # bool, optional; reject the shared
                                      # non-PE junk suffixes/substrings
                                      # (tarballs, skymaps, PESummaryTable, docs)
        dash_count: null             # int or null, optional; exact count of
                                      # "-" characters required in the name
        expected_event_count: null   # int or null, optional; sanity-check hint
        sample_set_policy: null      # str or null, optional; ingest hint

    metadata:                        # dict, required
      source_class_reference: ...    # str, required
      far_available_online: false   # bool, required
      preferred_waveform_source: ... # str, required

    validation:                      # dict, required
      require_checksums: true        # bool, required
      allow_missing_far: true        # bool, required

A file matches a product if it satisfies ALL of the checks that product
declares (checks with empty/false/null values are skipped).  See
``ProductSpec.matches``.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

try:  # Python >= 3.9
    from importlib import resources as importlib_resources
except ImportError:  # pragma: no cover
    import importlib_resources  # type: ignore

__all__ = [
    "ManifestValidationError",
    "RecordRef",
    "ProductSpec",
    "ReleaseManifest",
    "validate_manifest",
    "load_manifest_file",
    "list_release_manifests",
    "list_injection_manifests",
    "list_releases",
    "get_manifest",
]


class ManifestValidationError(ValueError):
    """A manifest is missing a required field or has a malformed value."""


# ---------------------------------------------------------------------------
# Shared, non-release-specific file-matching helpers
# ---------------------------------------------------------------------------
_JUNK_SUFFIXES = (".tar.gz", ".tar", ".ipynb", ".txt", ".md", ".fits", ".json")
_JUNK_SUBSTRINGS = ("PESummaryTable", "Skymap", "skymap", "Archived_Skymaps")
_EVENT_NAME_RE = re.compile(r"GW\d{6}")


def _is_junk(fn: str) -> bool:
    """True for non-PE files that live alongside PE samples in a record."""
    if any(fn.endswith(s) for s in _JUNK_SUFFIXES):
        return True
    if any(sub in fn for sub in _JUNK_SUBSTRINGS):
        return True
    return False


def _has_event_name(fn: str) -> bool:
    """True if the filename contains a GW event identifier."""
    return bool(_EVENT_NAME_RE.search(fn))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecordRef:
    """One provider record backing a release (e.g. one Zenodo deposit)."""
    provider: str
    record_id: int
    concept_id: Optional[int] = None


@dataclass(frozen=True)
class ProductSpec:
    """Declarative file-selection rule for one product within a release."""
    name: str
    file_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    contains_all: List[str] = field(default_factory=list)
    suffix: Optional[str] = None
    prefix: Optional[str] = None
    require_event_name: bool = False
    exclude_junk: bool = False
    dash_count: Optional[int] = None
    expected_event_count: Optional[int] = None
    sample_set_policy: Optional[str] = None

    def matches(self, filename: str) -> bool:
        """True if ``filename`` satisfies every check this product declares."""
        if self.exclude_junk and _is_junk(filename):
            return False
        if self.require_event_name and not _has_event_name(filename):
            return False
        if self.prefix and not filename.startswith(self.prefix):
            return False
        if self.suffix and not filename.endswith(self.suffix):
            return False
        if self.contains_all and not all(s in filename for s in self.contains_all):
            return False
        if self.exclude_patterns and any(
            fnmatch.fnmatch(filename, p) for p in self.exclude_patterns
        ):
            return False
        if self.dash_count is not None and filename.count("-") != self.dash_count:
            return False
        if self.file_patterns and not any(
            fnmatch.fnmatch(filename, p) for p in self.file_patterns
        ):
            return False
        return True


@dataclass(frozen=True)
class ReleaseManifest:
    """A fully parsed, validated release/injection manifest."""
    release: str
    observing_runs: List[str]
    description: str
    records: List[RecordRef]
    products: Dict[str, ProductSpec]
    metadata: dict
    validation: dict
    aliases: List[str] = field(default_factory=list)
    observing_run_label: Optional[str] = None
    source_path: str = "<unknown>"

    @property
    def observing_run(self) -> str:
        """Legacy single-string display, e.g. "O1+O2+O3a"."""
        return self.observing_run_label or "+".join(self.observing_runs)

    @property
    def record_ids(self) -> List[int]:
        return [r.record_id for r in self.records]

    @property
    def concept_ids(self) -> List[Optional[int]]:
        return [r.concept_id for r in self.records]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _require(raw: dict, key: str, types, source: str, where: str = ""):
    if key not in raw:
        raise ManifestValidationError(
            f"{source}: missing required field {where}{key!r}"
        )
    value = raw[key]
    if not isinstance(value, types):
        raise ManifestValidationError(
            f"{source}: field {where}{key!r} must be {types}, got {type(value)!r}"
        )
    return value


def _optional(raw: dict, key: str, types, default, source: str, where: str = ""):
    if key not in raw or raw[key] is None:
        return default
    value = raw[key]
    if not isinstance(value, types):
        raise ManifestValidationError(
            f"{source}: field {where}{key!r} must be {types} or null, "
            f"got {type(value)!r}"
        )
    return value


def validate_manifest(raw: dict, source: str = "<manifest>") -> None:
    """Validate a raw (post-YAML-parse) manifest dict.

    Raises ``ManifestValidationError`` naming the manifest file (``source``)
    and the offending field on any missing/malformed field or cross-field
    inconsistency. Does not mutate ``raw``.
    """
    if not isinstance(raw, dict):
        raise ManifestValidationError(
            f"{source}: manifest must be a YAML mapping, got {type(raw)!r}"
        )

    release = _require(raw, "release", str, source)
    if not release.strip():
        raise ManifestValidationError(f"{source}: field 'release' must be non-empty")

    observing_runs = _require(raw, "observing_runs", list, source)
    if not observing_runs or not all(isinstance(r, str) for r in observing_runs):
        raise ManifestValidationError(
            f"{source}: field 'observing_runs' must be a non-empty list of str"
        )

    _require(raw, "description", str, source)

    label = _optional(raw, "observing_run_label", str, None, source)

    aliases = _optional(raw, "aliases", list, [], source)
    if not all(isinstance(a, str) for a in aliases):
        raise ManifestValidationError(f"{source}: field 'aliases' must be list of str")
    if release in aliases:
        raise ManifestValidationError(
            f"{source}: field 'aliases' must not include the release's own "
            f"name {release!r}"
        )

    records = _require(raw, "records", list, source)
    if not records:
        raise ManifestValidationError(f"{source}: field 'records' must be non-empty")
    for i, rec in enumerate(records):
        where = f"records[{i}]."
        if not isinstance(rec, dict):
            raise ManifestValidationError(
                f"{source}: {where[:-1]} must be a mapping, got {type(rec)!r}"
            )
        _require(rec, "provider", str, source, where)
        record_id = _require(rec, "record_id", int, source, where)
        if isinstance(record_id, bool) or record_id <= 0:
            raise ManifestValidationError(
                f"{source}: field {where}'record_id' must be a positive int"
            )
        concept_id = _optional(rec, "concept_id", int, None, source, where)
        if concept_id is not None and (isinstance(concept_id, bool) or concept_id <= 0):
            raise ManifestValidationError(
                f"{source}: field {where}'concept_id' must be a positive int or null"
            )

    products = _require(raw, "products", dict, source)
    if not products:
        raise ManifestValidationError(f"{source}: field 'products' must be non-empty")
    for pname, pspec in products.items():
        where = f"products[{pname!r}]."
        if not isinstance(pspec, dict):
            raise ManifestValidationError(
                f"{source}: {where[:-1]} must be a mapping, got {type(pspec)!r}"
            )
        file_patterns = _require(pspec, "file_patterns", list, source, where)
        if not file_patterns or not all(isinstance(p, str) for p in file_patterns):
            raise ManifestValidationError(
                f"{source}: field {where}'file_patterns' must be a non-empty "
                "list of str"
            )
        exclude_patterns = _optional(pspec, "exclude_patterns", list, [], source, where)
        if not all(isinstance(p, str) for p in exclude_patterns):
            raise ManifestValidationError(
                f"{source}: field {where}'exclude_patterns' must be list of str"
            )
        contains_all = _optional(pspec, "contains_all", list, [], source, where)
        if not all(isinstance(p, str) for p in contains_all):
            raise ManifestValidationError(
                f"{source}: field {where}'contains_all' must be list of str"
            )
        _optional(pspec, "suffix", str, None, source, where)
        _optional(pspec, "prefix", str, None, source, where)
        _optional(pspec, "require_event_name", bool, False, source, where)
        _optional(pspec, "exclude_junk", bool, False, source, where)
        dash_count = _optional(pspec, "dash_count", int, None, source, where)
        if dash_count is not None and (isinstance(dash_count, bool) or dash_count < 0):
            raise ManifestValidationError(
                f"{source}: field {where}'dash_count' must be a non-negative int "
                "or null"
            )
        expected_event_count = _optional(
            pspec, "expected_event_count", int, None, source, where
        )
        if expected_event_count is not None and (
            isinstance(expected_event_count, bool) or expected_event_count < 0
        ):
            raise ManifestValidationError(
                f"{source}: field {where}'expected_event_count' must be a "
                "non-negative int or null"
            )
        _optional(pspec, "sample_set_policy", str, None, source, where)

    metadata = _require(raw, "metadata", dict, source)
    _require(metadata, "source_class_reference", str, source, "metadata.")
    _require(metadata, "far_available_online", bool, source, "metadata.")
    _require(metadata, "preferred_waveform_source", str, source, "metadata.")

    validation = _require(raw, "validation", dict, source)
    _require(validation, "require_checksums", bool, source, "validation.")
    _require(validation, "allow_missing_far", bool, source, "validation.")

    if label is not None and not label.strip():
        raise ManifestValidationError(
            f"{source}: field 'observing_run_label' must not be blank when present"
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _parse_manifest(raw: dict, source: str) -> ReleaseManifest:
    validate_manifest(raw, source=source)

    records = [
        RecordRef(
            provider=rec["provider"],
            record_id=rec["record_id"],
            concept_id=rec.get("concept_id"),
        )
        for rec in raw["records"]
    ]

    products = {}
    for pname, pspec in raw["products"].items():
        products[pname] = ProductSpec(
            name=pname,
            file_patterns=list(pspec["file_patterns"]),
            exclude_patterns=list(pspec.get("exclude_patterns") or []),
            contains_all=list(pspec.get("contains_all") or []),
            suffix=pspec.get("suffix"),
            prefix=pspec.get("prefix"),
            require_event_name=bool(pspec.get("require_event_name", False)),
            exclude_junk=bool(pspec.get("exclude_junk", False)),
            dash_count=pspec.get("dash_count"),
            expected_event_count=pspec.get("expected_event_count"),
            sample_set_policy=pspec.get("sample_set_policy"),
        )

    return ReleaseManifest(
        release=raw["release"],
        observing_runs=list(raw["observing_runs"]),
        observing_run_label=raw.get("observing_run_label"),
        description=raw["description"],
        aliases=list(raw.get("aliases") or []),
        records=records,
        products=products,
        metadata=dict(raw["metadata"]),
        validation=dict(raw["validation"]),
        source_path=source,
    )


def load_manifest_file(path) -> ReleaseManifest:
    """Load and validate a single manifest YAML file from a filesystem path."""
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ManifestValidationError(f"{path}: manifest file is empty")
    return _parse_manifest(raw, source=str(path))


# ---------------------------------------------------------------------------
# Bundled manifest discovery
# ---------------------------------------------------------------------------
def _bundled_yaml_paths(subdir: str) -> List[Path]:
    pkg_dir = importlib_resources.files(__name__) / subdir
    with importlib_resources.as_file(pkg_dir) as p:
        return sorted(Path(p).glob("*.yaml")) + sorted(Path(p).glob("*.yml"))


def _load_bundled(subdir: str) -> Dict[str, ReleaseManifest]:
    """Load every manifest in a bundled subdirectory, keyed by release name
    (canonical name and aliases both resolve to the same object)."""
    manifests: Dict[str, ReleaseManifest] = {}
    for path in _bundled_yaml_paths(subdir):
        manifest = load_manifest_file(path)
        if manifest.release in manifests:
            raise ManifestValidationError(
                f"{path}: duplicate release name {manifest.release!r} "
                f"(already defined in {manifests[manifest.release].source_path})"
            )
        manifests[manifest.release] = manifest
        for alias in manifest.aliases:
            if alias in manifests:
                raise ManifestValidationError(
                    f"{path}: alias {alias!r} collides with an existing "
                    "release/alias name"
                )
            manifests[alias] = manifest
    return manifests


def _load_all_bundled() -> Dict[str, ReleaseManifest]:
    """Load every bundled manifest (releases/ + injections/), keyed by
    canonical release name and alias, with cross-directory collision checks."""
    releases = _load_bundled("releases")
    injections = _load_bundled("injections")
    collisions = set(releases) & set(injections)
    if collisions:
        raise ManifestValidationError(
            f"Release name(s) {sorted(collisions)} defined in both "
            "manifests/releases/ and manifests/injections/"
        )
    merged = dict(releases)
    merged.update(injections)
    return merged


def list_release_manifests() -> List[str]:
    """Canonical names of bundled PE release manifests (aliases excluded)."""
    manifests = _load_bundled("releases")
    return sorted({m.release for m in manifests.values()})


def list_injection_manifests() -> List[str]:
    """Canonical names of bundled injection manifests (aliases excluded)."""
    manifests = _load_bundled("injections")
    return sorted({m.release for m in manifests.values()})


def list_releases() -> List[str]:
    """Canonical names of every bundled manifest (releases + injections)."""
    return sorted(set(list_release_manifests()) | set(list_injection_manifests()))


def get_manifest(name: str) -> ReleaseManifest:
    """Look up a manifest by bundled name/alias, or load one from a path.

    If ``name`` resolves to an existing file on disk, it is loaded directly
    (this is how user-supplied manifests are added without any code changes).
    Otherwise ``name`` is looked up among the bundled release and injection
    manifests (canonical names and aliases).
    """
    candidate = Path(name)
    if candidate.is_file():
        return load_manifest_file(candidate)

    bundled = _load_all_bundled()
    if name in bundled:
        return bundled[name]

    available = sorted(set(bundled))
    raise KeyError(f"Unknown manifest {name!r}. Available: {available}")
