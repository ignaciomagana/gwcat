"""The explicit release-reweight cosmology table (GW-40a).

GWTC-2.1 and GWTC-3 ``*_cosmo.h5`` posteriors were reweighted by the LVK to a
luminosity-distance prior uniform in comoving volume and source-frame time
(UniformSourceFrame), but no field of those files reliably says at WHICH
cosmology:

* ``priors/analytic/luminosity_distance`` is the stale ``PowerLaw(alpha=2)``
  sampling prior and names no cosmology;
* ``meta_data/meta_data/cosmology`` is absent on every ``C01:Mixed`` set and
  contradicts the stored z(dL) for several later labels.

Before GW-40a gwcat evaluated these rows at ``IngestConfig.o3_default_cosmo``
(astropy Planck15, 67.74/0.3075), a code default chosen by the label PREFIX
``C01``.  That was never a statement from any release, and the prefix is not the
catalog (GWTC-5's GW240925_005809 is released with ``C01`` labels).

This module replaces it with one explicit, cited table keyed by CATALOG
(``gwcat/data/release_reweight_cosmology.yaml``).  Every store row it resolves
records the row's cosmology name, its ``source`` (``documented`` or
``inferred_from_z(dL)``) and the table file's sha256, so the choice is
auditable and a different table is a visible, deliberate input -- never a
silent default.  There is no fallback: a release-reweighted row whose catalog
the table does not list is refused.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

#: The bundled production table (operator decision OD-2).
DEFAULT_TABLE_FILENAME = "release_reweight_cosmology.yaml"
#: The bundled pre-GW-40 behaviour (astropy Planck15), for regressions ONLY.
LEGACY_TABLE_FILENAME = "release_reweight_cosmology_legacy_astropy.yaml"

#: ``source`` values a production row may carry.
PRODUCTION_SOURCES = ("documented", "inferred_from_z(dL)")
#: Every ``source`` value the loader accepts (the legacy table's included).
ALLOWED_SOURCES = PRODUCTION_SOURCES + ("legacy_gwcat_default",)


class ReleaseReweightCosmologyError(ValueError):
    """The release-reweight cosmology table is malformed or lacks a catalog."""


@dataclass(frozen=True)
class ReleaseCosmologyRow:
    """One catalog's reweighting cosmology, with where it came from."""
    catalog: str
    name: str
    H0: float
    Om0: float
    source: str
    citation: str
    corroboration: str = ""
    #: The release record stating the reweighted prior's form (optional).
    record_citation: str = ""


@dataclass(frozen=True)
class ReleaseCosmologyTable:
    """A loaded table: its rows plus the identity of the file they came from."""
    rows: Dict[str, ReleaseCosmologyRow]
    path: str
    sha256: str
    table_id: str = ""

    def lookup(self, catalog: str) -> ReleaseCosmologyRow:
        """The row for ``catalog``; refuses (never guesses) when absent."""
        try:
            return self.rows[catalog]
        except KeyError:
            raise ReleaseReweightCosmologyError(
                f"catalog {catalog!r} is release-reweighted (a *_cosmo file) "
                f"but the release-reweight cosmology table {self.path} "
                f"(sha256 {self.sha256[:12]}...) has no row for it; it lists "
                f"{sorted(self.rows)}. gwcat will not fall back to a default "
                f"cosmology for a reweighted release -- add a cited row for "
                f"{catalog!r} to the table (or pass a table that has one)."
            ) from None


def bundled_table_path(filename: str = DEFAULT_TABLE_FILENAME) -> str:
    """Filesystem path of a table bundled under ``gwcat/data``."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        filename)


_CACHE: Dict[str, ReleaseCosmologyTable] = {}


def load_release_cosmology_table(path: Optional[str] = None
                                 ) -> ReleaseCosmologyTable:
    """Load and validate a release-reweight cosmology table.

    ``path=None`` loads the bundled production table.  The sha256 is of the
    file's exact bytes, so two tables with the same numbers but different
    citations are distinguishable in the provenance.
    """
    import yaml

    path = os.path.abspath(path or bundled_table_path())
    with open(path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    cached = _CACHE.get(path)
    if cached is not None and cached.sha256 == sha:
        return cached

    doc = yaml.safe_load(raw.decode("utf-8")) or {}
    cats = doc.get("catalogs")
    if not isinstance(cats, dict) or not cats:
        raise ReleaseReweightCosmologyError(
            f"{path}: no 'catalogs' mapping in the release-reweight cosmology "
            f"table.")
    rows = {}
    for cat, row in cats.items():
        if not isinstance(row, dict):
            raise ReleaseReweightCosmologyError(
                f"{path}: row {cat!r} is not a mapping.")
        missing = [k for k in ("name", "H0", "Om0", "source", "citation")
                   if k not in row or row[k] in (None, "")]
        if missing:
            raise ReleaseReweightCosmologyError(
                f"{path}: row {cat!r} lacks {missing}; every row must state "
                f"its cosmology AND where that value comes from.")
        H0, Om0 = float(row["H0"]), float(row["Om0"])
        if not (math.isfinite(H0) and H0 > 0 and math.isfinite(Om0)
                and 0 < Om0 < 1):
            raise ReleaseReweightCosmologyError(
                f"{path}: row {cat!r} has unphysical (H0, Om0) = "
                f"({H0!r}, {Om0!r}).")
        src = str(row["source"])
        if src not in ALLOWED_SOURCES:
            raise ReleaseReweightCosmologyError(
                f"{path}: row {cat!r} has source={src!r}; allowed: "
                f"{list(ALLOWED_SOURCES)}.")
        rows[str(cat)] = ReleaseCosmologyRow(
            catalog=str(cat), name=str(row["name"]), H0=H0, Om0=Om0,
            source=src, citation=" ".join(str(row["citation"]).split()),
            corroboration=" ".join(str(row.get("corroboration", "")).split()),
            record_citation=" ".join(
                str(row.get("record_citation", "")).split()))
    table = ReleaseCosmologyTable(rows=rows, path=path, sha256=sha,
                                  table_id=str(doc.get("table_id", "")))
    _CACHE[path] = table
    return table
