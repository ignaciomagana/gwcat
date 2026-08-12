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
)

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
    return hashlib.blake2b(payload.encode(),
                           digest_size=HASH_LEN // 2).hexdigest()


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
