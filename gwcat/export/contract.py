"""The pairing contract and its hash (GW-22, design PR-G).

A PE export and a selection export are only meaningfully paired if they agree
about *what was fitted and under what assumptions*.  Today that agreement is
checked field by field in the validator, which means every new field is a new
check somebody has to remember to add.  A ``contract_hash`` makes the agreement
one comparison, and -- more importantly -- makes a DISAGREEMENT detectable on a
file that looks entirely healthy.

What is in the hash, and why each item
--------------------------------------
* ``parameter_space`` / ``fit_columns`` / ``advisory_columns`` -- the coordinates
  the density covers.  A component PE file paired with a chieff selection file is
  the central contract error; this is what catches it.
* ``spin_basis_kind`` (projection vs bijection) and ``spin_density_exact`` -- the
  R1 contract.  A projection paired with a campaign it cannot describe is wrong
  even when both files individually validate.
* ``mass_prior_kind`` and ``dL_prior_kind`` -- the parsed prior CLASSES (GW-02,
  GW-07).  Two files built against different distance-prior classes are not
  comparable, and until GW-02 nothing recorded the class at all.
* the detection cut (``far_max`` / ``snr_min`` / ``allow_missing_far``) -- Essick
  & Fishbach's central requirement is that the event cut and the injection
  detection cut are the same statistic at the same threshold.  This is the one
  the validator could not check before, because the numeric threshold was never
  exported.
* canonical ``source_class`` -- a filtered PE file paired with an unfiltered
  selection file is a silent bias.
* ``cosmology`` -- and note the consumer reads ``pe_cosmology_H0``/``Om0`` from
  no code path at all, so this hash is the only place the agreement is enforced.

What is deliberately NOT in the hash
------------------------------------
The PE and selection ``amax`` legitimately DIFFER: the PE side removes the
posterior's own spin prior (ceiling ~0.99) while the selection side swaps the
injected draw (endo3 injects 0.998).  Requiring them equal would force the
injection campaign's ceiling to match the PE prior's, which need not hold.  That
stays a warn-not-fail cross-check.
"""
from __future__ import annotations

import hashlib
import json

#: Fields, in a fixed order, that constitute the pairing contract.
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
    """A stable short digest of a contract dict."""
    payload = json.dumps({k: contract.get(k) for k in CONTRACT_FIELDS},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode(),
                           digest_size=HASH_LEN // 2).hexdigest()


def contract_diff(pe_contract: dict, sel_contract: dict) -> list:
    """Field-by-field differences, so a hash mismatch is actionable.

    A bare "hashes differ" is useless to an operator; this is what turns it into
    "these two files disagree about the distance-prior class".
    """
    out = []
    for k in CONTRACT_FIELDS:
        a, b = pe_contract.get(k), sel_contract.get(k)
        if a != b:
            out.append((k, a, b))
    return out


def format_diff(diff) -> str:
    if not diff:
        return "(no field-level differences)"
    return "; ".join(f"{k}: PE={a!r} vs selection={b!r}" for k, a, b in diff)
