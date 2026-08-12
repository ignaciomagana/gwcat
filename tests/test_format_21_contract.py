"""Format 2.1 and the pairing contract hash (GW-22, design PR-G).

2.1 is deliberately a SEPARATE registered format rather than a bump of
``gwcat2``: ``_require_hdf5_format`` in the consumer is a closed literal tuple,
so a 2.1 file is unloadable by any darksirens that has not been taught the
version, and the plan's rule is that the consumer patch deploys FIRST.  These
tests pin that 2.0 stays the default, so nothing gwcat writes by accident
becomes unreadable downstream.

The contract hash exists to turn a growing list of per-field pairing checks into
one comparison -- and, more importantly, to make a DISAGREEMENT detectable on a
pair of files that each validate perfectly on their own.
"""
import json

import numpy as np
import h5py
import pytest

from gwcat.export.contract import (CONTRACT_FIELDS, build_contract,
                                   contract_diff, contract_hash, format_diff)


# --------------------------------------------------------------------------
# 1. The hash itself
# --------------------------------------------------------------------------
def test_hash_is_stable_and_order_independent():
    a = build_contract(parameter_space="component", fit_columns=["a", "b"],
                       cosmology_H0=67.74)
    b = build_contract(cosmology_H0=67.74, fit_columns=["a", "b"],
                       parameter_space="component")
    assert contract_hash(a) == contract_hash(b)
    assert len(contract_hash(a)) == 16


def test_hash_changes_when_any_contract_field_changes():
    base = build_contract(parameter_space="component")
    h0 = contract_hash(base)
    for field in CONTRACT_FIELDS:
        other = build_contract(**{**{"parameter_space": "component"},
                                  field: "PERTURBED"})
        if field == "parameter_space":
            continue
        assert contract_hash(other) != h0, f"{field} does not affect the hash"


def test_unknown_contract_field_is_rejected():
    """A key the hash silently ignores is a check that silently stops running."""
    with pytest.raises(ValueError, match="unknown contract field"):
        build_contract(parameter_space="component", nonsense=1)


def test_float_rounding_cannot_flip_the_hash():
    """An HDF5 round-trip must not change the digest."""
    a = build_contract(cosmology_H0=67.74)
    b = build_contract(cosmology_H0=67.74 + 1e-15)
    assert contract_hash(a) == contract_hash(b)


def test_diff_names_the_disagreeing_field():
    """A bare "hashes differ" is useless to an operator."""
    pe = build_contract(parameter_space="component", dL_prior_kind="PowerLaw")
    sel = build_contract(parameter_space="component",
                         dL_prior_kind="UniformSourceFrame")
    d = contract_diff(pe, sel)
    assert len(d) == 1 and d[0][0] == "dL_prior_kind"
    assert "dL_prior_kind" in format_diff(d)
    assert contract_diff(pe, pe) == []


def test_the_detection_cut_is_in_the_contract():
    """Essick & Fishbach's central requirement: the event cut and the injection
    detection cut must be the same statistic at the same threshold.  Until the
    numeric threshold was exported, that check could not be made at all."""
    assert "detection_statistic" in CONTRACT_FIELDS
    assert "detection_threshold" in CONTRACT_FIELDS
    a = build_contract(detection_statistic="far_max", detection_threshold=1.0)
    b = build_contract(detection_statistic="far_max", detection_threshold=2.0)
    assert contract_hash(a) != contract_hash(b)
    # same threshold, different STATISTIC is also a mismatch
    c = build_contract(detection_statistic="snr_min", detection_threshold=1.0)
    assert contract_hash(a) != contract_hash(c)


def test_amax_is_deliberately_not_in_the_contract():
    """The PE and selection amax legitimately DIFFER -- the PE side removes the
    posterior's own spin prior (~0.99), the selection side swaps the injected
    draw (endo3 injects 0.998).  Requiring them equal would force the campaign's
    ceiling to match the PE prior's, which need not hold."""
    assert not any("amax" in f for f in CONTRACT_FIELDS)


# --------------------------------------------------------------------------
# 2. The 2.1 writers, against the real re-ingested store
# --------------------------------------------------------------------------
STORE = "working/gw16/gwcat_store_all_gwtc_v13.h5"


def _have_store():
    import os
    return os.path.exists(STORE)


needs_store = pytest.mark.skipif(not _have_store(),
                                 reason="requires the GW-16 re-ingested store")


def _export(tmp_path, fmt, basis, name):
    from gwcat.catalog import GWCatalog

    cat = GWCatalog(STORE)
    names = list(cat.event_names[:3])
    out = tmp_path / name
    cat.export(str(out), format=fmt, spin_basis=basis, nsamp=64, seed=0,
               cosmology=(67.74, 0.3089), allowed_names=names,
               allowed_names_authoritative=True)
    with h5py.File(out, "r") as f:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v)
                 for k, v in f.attrs.items()}
    return attrs


@needs_store
def test_default_format_is_still_2_0(tmp_path):
    """The lockstep guarantee: nothing becomes unloadable by accident."""
    a = _export(tmp_path, "gwcat2", "chieff", "d.h5")
    assert a["format_version"] == "gwcat-pe-2.0"
    assert "contract_hash" not in a


@needs_store
def test_2_1_carries_the_declared_contract(tmp_path):
    a = _export(tmp_path, "gwcat2.1", "component", "c.h5")
    assert a["format_version"] == "gwcat-pe-2.1"
    assert a["parameter_space"] == "component"
    assert a["spin_basis_kind"] == "bijective"
    assert bool(a["spin_density_exact"]) is True
    fit = [x.decode() if isinstance(x, bytes) else x for x in a["fit_columns"]]
    adv = [x.decode() if isinstance(x, bytes) else x
           for x in a["advisory_columns"]]
    # the advisory hole, now stated ON THE FILE rather than only in the registry
    assert "chip" in adv and "chip" not in fit
    assert "a1" in fit
    blocks = json.loads(a["block_provenance"])
    assert blocks["space"] == "component"
    assert len(blocks["block_provenance"]) == len(blocks["blocks"])
    json.loads(a["contract"])
    assert len(a["contract_hash"]) == 16


@needs_store
def test_a_projection_file_says_so_on_its_face(tmp_path):
    a = _export(tmp_path, "gwcat2.1", "chieff", "p.h5")
    assert a["spin_basis_kind"] == "projection"
    assert bool(a["spin_density_exact"]) is False


@needs_store
def test_different_spaces_get_different_contract_hashes(tmp_path):
    """A component PE file paired with a chieff selection file is the central
    contract error; distinct hashes are what make it one comparison."""
    a = _export(tmp_path, "gwcat2.1", "chieff", "h1.h5")
    b = _export(tmp_path, "gwcat2.1", "component", "h2.h5")
    assert a["contract_hash"] != b["contract_hash"]


@needs_store
def test_the_same_space_reproduces_the_same_hash(tmp_path):
    a = _export(tmp_path, "gwcat2.1", "component", "s1.h5")
    b = _export(tmp_path, "gwcat2.1", "component", "s2.h5")
    assert a["contract_hash"] == b["contract_hash"]


def test_both_2_1_writers_are_registered():
    from gwcat.export.registry import list_formats

    fmts = set(list_formats())
    assert ("gwcat2.1", "pe") in fmts
    assert ("gwcat2.1", "selection") in fmts
    # and 2.0 is still there, unchanged
    assert ("gwcat2", "pe") in fmts
    assert ("gwcat2", "selection") in fmts
