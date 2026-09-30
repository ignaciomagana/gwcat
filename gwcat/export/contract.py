"""The pairing contract and its hash (GW-22, design PR-G).

A PE export and a selection export are only meaningfully paired if they agree
about *what was fitted and under what assumptions*.  Today that agreement is
checked field by field in the validator, which means every new field is a new
check somebody has to remember to add.  A ``contract_hash`` makes the agreement
one comparison, and -- more importantly -- makes a DISAGREEMENT detectable on a
file that looks entirely healthy.

The contract, and the subset that is hashed
-------------------------------------------
``CONTRACT_FIELDS`` is the full declaration, written to both files and used to
produce a field-by-field diff.  ``PAIRING_FIELDS`` is the subset the **hash**
covers: the declarations that must be *byte-identical* on the two sides for the
pairing to mean anything.

* ``parameter_space`` / ``fit_columns`` / ``advisory_columns`` -- the coordinates
  the density covers.  A component PE file paired with a chieff selection file is
  the central contract error; this is what catches it.
* ``spin_basis_kind`` (projection vs bijection) and ``spin_density_exact`` -- the
  R1 contract.  A projection paired with a campaign it cannot describe is wrong
  even when both files individually validate.
* ``sky_prior_in_density`` -- whether the sky measure is inside the density, which
  must be the same convention on both sides or it does not cancel.
* canonical ``source_class`` -- a filtered PE file paired with an unfiltered
  selection file is a silent bias.

The effective event selection (GW-33)
-------------------------------------
The class filter was for a long time the ONLY event cut in the contract, so a PE
file could state a matching ``source_class`` while having been cut on p_astro, on
a name whitelist, on median source-frame masses or on SNR -- none of which the
paired injection campaign reproduces, and none of which anything compared.  The
contract therefore also carries every OTHER effective cut
(:data:`SELECTION_CONTRACT_FIELDS`), plus two deterministic digests: one over the
name whitelist that was requested and one over the event list actually written.
They are recorded and diffed rather than hashed -- an injection campaign has no
event list to digest, so an equality hash over them could never match.

Why the rest are recorded but NOT hashed
----------------------------------------
An equality hash is the wrong instrument for a field whose two sides are
legitimately different, or equal only to a tolerance.  Hashing those would make
the digest differ on every correct pair, which is how a check stops being read.

* ``mass_prior_kind`` / ``dL_prior_kind`` describe the *PE prior*; an injection
  campaign has no such prior to declare, so the selection side records ``None``
  and the hash could never match.  They are compared PE-to-PE and carried in the
  diff.
* the detection cut (``detection_statistic`` / ``detection_threshold`` /
  ``allow_missing_far``) is Essick & Fishbach's central requirement, but the
  shipped configuration selects events by *name whitelist* against a FAR-cut
  injection set -- a correct pairing in which the two sides state different
  things.  ``validate._xcheck_detection_cut`` encodes that nuance: fail on two
  stated thresholds that differ, warn when only one side states one.
* ``cosmology_H0`` / ``cosmology_Om0`` are compared with the physical tolerances
  ``|dH0| >= 1.0`` / ``|dOm0| >= 0.05``, which an exact digest cannot express.

The PE and selection ``amax`` are in neither: they legitimately DIFFER, because
the PE side removes the posterior's own spin prior (ceiling ~0.99) while the
selection side swaps the injected draw (endo3 injects 0.998).  That stays a
warn-not-fail cross-check.
"""
from __future__ import annotations

import hashlib
import json

#: The effective-event-selection fields (GW-33), read off a product's
#: ``selection_spec`` provenance by :func:`selection_contract_fields`.  Every
#: cut a view accumulated, so a filtered PE file cannot present itself as
#: unfiltered, plus the two digests that identify WHICH events it holds.
SELECTION_CONTRACT_FIELDS = (
    "compact_type",
    # The PE export's per-sample redshift truncation, and the injection subset
    # that matches it (GW-37).  It was in no contract field at all, so a
    # truncated PE product and the full one it came from hashed and diffed
    # identically while covering different redshift ranges.
    "z_max",
    "pastro_min",
    "snr_min",
    "sky_area_max",
    "m1_src_range",
    "m2_src_range",
    "allowed_names_digest",
    "event_list_digest",
    "selection_spec_digest",
)

#: Fields, in a fixed order, that constitute the full declared contract.
CONTRACT_FIELDS = (
    "parameter_space",
    "fit_columns",
    "advisory_columns",
    "spin_basis_kind",
    "spin_density_exact",
    "mass_prior_kind",
    "dL_prior_kind",
    "sky_prior_in_density",
    "source_class",
    "detection_statistic",
    "detection_threshold",
    "allow_missing_far",
    "cosmology_H0",
    "cosmology_Om0",
) + SELECTION_CONTRACT_FIELDS

#: The subset the hash covers -- the declarations that must match EXACTLY across
#: a PE/selection pair.  See the module docstring for why each of the others is
#: recorded and diffed but checked by a purpose-built comparison instead.
PAIRING_FIELDS = (
    "parameter_space",
    "fit_columns",
    "advisory_columns",
    "spin_basis_kind",
    "spin_density_exact",
    "sky_prior_in_density",
    "source_class",
)

#: Length of the hex digest.  16 hex chars = 64 bits: ample for detecting an
#: accidental mismatch, and short enough to read in an error message.
HASH_LEN = 16


def _canonical(value):
    """A stable, JSON-able form for one contract field."""
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, float):
        # Round so float round-trip through HDF5 cannot flip the hash.
        return round(float(value), 10)
    if value is None:
        return None
    if isinstance(value, (int, str)):
        return value
    return str(value)


def stable_digest(payload: str) -> str:
    """The package's one short-digest convention, over an already-canonical string.

    Shared by :func:`contract_hash`, :func:`event_list_digest` and
    :meth:`gwcat.catalog.SelectionSpec.digest` so two digests of the same thing
    cannot come out different because two call sites picked two hash functions.
    """
    return hashlib.blake2b(payload.encode(),
                           digest_size=HASH_LEN // 2).hexdigest()


def event_list_digest(names) -> str:
    """A deterministic digest over a SET of event names.

    Sorted and de-duplicated first, so the digest identifies *which events* a
    file holds and not the order they happened to be written in: two exports of
    the same event list agree, and one extra or missing event does not.  This is
    what lets a contract state its event selection without carrying (and
    round-tripping) the whole list.

    ``None`` -- "no list recorded" -- returns ``""``, which is distinct from the
    digest of the empty list.
    """
    if names is None:
        return ""
    uniq = sorted({(n.decode() if isinstance(n, (bytes, bytearray)) else str(n))
                   for n in names})
    return stable_digest(json.dumps(uniq, separators=(",", ":")))


def selection_contract_fields(attrs) -> dict:
    """The :data:`SELECTION_CONTRACT_FIELDS` of a product, from its attrs.

    Reads the ``selection_spec`` JSON the builders write (the EFFECTIVE cuts,
    accumulated across every ``select()`` the view went through) rather than
    re-deriving them from the export call's arguments -- re-deriving is exactly
    how a filtered file came to advertise itself as unfiltered.  A product that
    states no spec (the selection/injection side, or a file written before
    GW-33) yields all-``None``, which the diff reports as "not stated".
    """
    out = {k: None for k in SELECTION_CONTRACT_FIELDS}

    def _text(v):
        if v is None:
            return None
        if isinstance(v, (bytes, bytearray)):
            v = v.decode()
        s = str(v).strip()
        return s or None

    raw = _text(attrs.get("selection_spec"))
    try:
        spec = json.loads(raw) if raw else None
    except ValueError:
        spec = None
    if not spec:
        return out

    out.update({
        "compact_type": list(spec.get("compact_type") or ()) or None,
        "pastro_min": spec.get("pastro_min"),
        "snr_min": spec.get("snr_min"),
        "sky_area_max": spec.get("sky_area_max"),
        "m1_src_range": spec.get("m1_src_range"),
        "m2_src_range": spec.get("m2_src_range"),
        "allowed_names_digest": _text(spec.get("allowed_names_digest")),
        "event_list_digest": _text(attrs.get("event_list_digest")),
        "selection_spec_digest": _text(attrs.get("selection_spec_digest")),
    })
    return out


def build_contract(**fields) -> dict:
    """Assemble a contract dict, filling absent fields with ``None``.

    Unknown keys raise: a field silently dropped from the hash is a field the
    pairing check stops covering, which is exactly the failure mode this exists
    to prevent.
    """
    unknown = set(fields) - set(CONTRACT_FIELDS)
    if unknown:
        raise ValueError(
            f"unknown contract field(s) {sorted(unknown)}; the contract is "
            f"{list(CONTRACT_FIELDS)}. Add the field there deliberately -- a key "
            f"the hash silently ignores is a check that silently stops running.")
    return {k: _canonical(fields.get(k)) for k in CONTRACT_FIELDS}


def contract_hash(contract: dict) -> str:
    """A stable short digest over :data:`PAIRING_FIELDS`.

    Deliberately not over every field: the digest exists to be compared between
    a PE file and its selection file, so a field the two sides legitimately
    state differently would make it mismatch on every correct pair.
    """
    payload = json.dumps({k: contract.get(k) for k in PAIRING_FIELDS},
                         sort_keys=True, separators=(",", ":"))
    return stable_digest(payload)


def contract_diff(pe_contract: dict, sel_contract: dict, fields=None) -> list:
    """Field-by-field differences, so a hash mismatch is actionable.

    A bare "hashes differ" is useless to an operator; this is what turns it into
    "these two files disagree about the distance-prior class".  Pass
    ``fields=PAIRING_FIELDS`` to explain a hash mismatch specifically -- over
    the full field set the answer would be buried among the fields the two sides
    are *expected* to state differently.
    """
    out = []
    for k in (CONTRACT_FIELDS if fields is None else fields):
        a, b = pe_contract.get(k), sel_contract.get(k)
        if a != b:
            out.append((k, a, b))
    return out


def format_diff(diff) -> str:
    if not diff:
        return "(no field-level differences)"
    return "; ".join(f"{k}: PE={a!r} vs selection={b!r}" for k, a, b in diff)


# ---------------------------------------------------------------------------
# Prior-implementation families (GW-40i)
# ---------------------------------------------------------------------------
#: ``dL_prior_impl`` values that evaluate the declared analytic distance prior
#: to rounding: the exact UniformSourceFrame/UniformComovingVolume density, and
#: the closed-form classes (``"analytic"``), which have no implementation choice.
DL_PRIOR_IMPLS_EXACT = ("exact", "analytic")
#: The pre-GW-40i interpolated UniformSourceFrame implementations.
DL_PRIOR_IMPLS_LEGACY = ("bilby", "astropy")


def _impl_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        x = x.decode()
    return "" if x is None else str(x)


def mixed_prior_impl_events(chi_eff_impl, dL_impls, names) -> list:
    """Events whose ``p_dL_pe`` implementation is of the OTHER family than the
    product's chi_eff implementation.

    ``chi_eff_impl == "exact"`` flags events whose distance prior was evaluated
    by a legacy interpolation (``"bilby"``/``"astropy"``); ``"grid"`` flags
    events evaluated ``"exact"``.  ``"analytic"`` rows (a closed-form class,
    identical in both families) and ``""`` (unrecorded) are never flagged.
    """
    chi = _impl_str(chi_eff_impl)
    if chi == "exact":
        other = DL_PRIOR_IMPLS_LEGACY
    elif chi == "grid":
        other = ("exact",)
    else:
        return []
    return [str(_impl_str(n)) for n, d in zip(names, dL_impls)
            if _impl_str(d) in other]
