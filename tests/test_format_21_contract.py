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

from gwcat.export.contract import (CONTRACT_FIELDS, PAIRING_FIELDS,
                                   build_contract, contract_diff,
                                   contract_hash, format_diff)


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


def test_hash_changes_when_any_pairing_field_changes():
    base = build_contract(parameter_space="component")
    h0 = contract_hash(base)
    for field in PAIRING_FIELDS:
        other = build_contract(**{**{"parameter_space": "component"},
                                  field: "PERTURBED"})
        if field == "parameter_space":
            continue
        assert contract_hash(other) != h0, f"{field} does not affect the hash"


def test_hash_ignores_the_fields_the_two_sides_state_differently():
    """The digest is compared ACROSS a pair, so it may only cover fields both
    sides can state identically.

    ``mass_prior_kind``/``dL_prior_kind`` describe the PE prior and an injection
    campaign has none; the detection cut is legitimately stated on one side only
    (a name-whitelisted event list against a FAR-cut injection set); and the
    cosmology is compared to a tolerance an exact digest cannot express.  Hashing
    those made the digest differ on every correct pair -- which went unnoticed
    because nothing compared a PE hash to a selection hash until GW-11.
    """
    base = build_contract(parameter_space="component")
    h0 = contract_hash(base)
    for field in set(CONTRACT_FIELDS) - set(PAIRING_FIELDS):
        other = build_contract(**{**{"parameter_space": "component"},
                                  field: "PERTURBED"})
        assert contract_hash(other) == h0, f"{field} must not affect the hash"
        # ...but it is still recorded, and still diffed.
        assert field in CONTRACT_FIELDS
        assert contract_diff(base, other)


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
    numeric threshold was exported, that check could not be made at all.

    It is recorded and diffed, but compared by ``_xcheck_detection_cut`` rather
    than by the digest -- the shipped configuration states the cut on one side
    only, and an equality hash cannot express "fail on two stated thresholds
    that differ, warn when only one side states one".
    """
    assert "detection_statistic" in CONTRACT_FIELDS
    assert "detection_threshold" in CONTRACT_FIELDS
    a = build_contract(detection_statistic="far", detection_threshold=1.0)
    b = build_contract(detection_statistic="far", detection_threshold=2.0)
    assert contract_diff(a, b) == [("detection_threshold", 1.0, 2.0)]


def test_every_effective_event_cut_is_in_the_contract():
    """The class filter used to be the ONLY event cut in the contract, so a PE
    file cut on p_astro, on a name whitelist, on median masses or on SNR could
    state a matching ``source_class`` and pass the pairing check (GW-33)."""
    for field in ("pastro_min", "snr_min", "sky_area_max", "compact_type",
                  "m1_src_range", "m2_src_range", "allowed_names_digest",
                  "event_list_digest", "selection_spec_digest"):
        assert field in CONTRACT_FIELDS, field
        # Recorded and diffed, not hashed: an injection campaign has no event
        # list to digest and no p_astro to threshold, so an equality hash over
        # these could never match on a correct pair.
        assert field not in PAIRING_FIELDS, field
    a = build_contract(pastro_min=None)
    b = build_contract(pastro_min=0.9)
    assert contract_diff(a, b) == [("pastro_min", None, 0.9)]


def test_the_contract_reads_the_effective_selection_off_a_product():
    """It reads the builders' ``selection_spec`` -- the EFFECTIVE cuts -- rather
    than re-deriving them from the export call, which is how a filtered file
    came to describe itself as unfiltered."""
    from gwcat.catalog import SelectionSpec
    from gwcat.export.contract import selection_contract_fields

    spec = SelectionSpec().refine(source_class="bbh", pastro_min=0.9,
                                  m1_src_range=(5.0, 100.0))
    attrs = dict(spec.to_attrs())
    attrs["event_list_digest"] = "abc123"
    fields = selection_contract_fields(attrs)
    assert fields["pastro_min"] == 0.9
    assert fields["m1_src_range"] == [5.0, 100.0]
    assert fields["event_list_digest"] == "abc123"
    assert fields["selection_spec_digest"] == spec.digest()
    # A product that states no spec (the selection side, or a pre-GW-33 file)
    # says "not stated" rather than "no cuts".
    assert all(v is None for v in selection_contract_fields({}).values())
    # ...and the contract accepts every one of them by name.
    build_contract(**selection_contract_fields(attrs))


def test_event_list_digest_identifies_which_events_not_their_order():
    from gwcat.export.contract import event_list_digest

    names = ["GW150914", "GW170817", "GW190521"]
    assert event_list_digest(names) == event_list_digest(names[::-1])
    assert event_list_digest(names) == event_list_digest(names + names)
    assert event_list_digest(names) != event_list_digest(names[:2])
    assert event_list_digest([b"GW150914"]) == event_list_digest(["GW150914"])
    # "no list recorded" is not the digest of the empty list
    assert event_list_digest(None) == ""
    assert event_list_digest([]) != ""
    assert len(event_list_digest(names)) == 16


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
def test_2_1_declares_the_coordinates_its_density_is_actually_in(tmp_path):
    """``p_pe`` carries ``m1det``, which is ``|d(m1det,m2det)/d(m1det,q)|`` --
    the Jacobian for ``(m1det, q)``.  The contract nevertheless published
    ``(m1det, m2det)`` as its fit columns, so a generic consumer reading the
    contract off the file could apply the wrong measure and nothing in the file
    would contradict it.
    """
    a = _export(tmp_path, "gwcat2.1", "component", "coord.h5")
    fit = [x.decode() if isinstance(x, bytes) else x for x in a["fit_columns"]]
    adv = [x.decode() if isinstance(x, bytes) else x
           for x in a["advisory_columns"]]
    assert "q" in fit and "m2det" not in fit
    assert "m2det" in adv
    assert a["mass_density_coordinates"] == "m1det,q"
    contract = json.loads(a["contract"])
    assert contract["fit_columns"] == fit
    assert contract["advisory_columns"] == adv


@needs_store
def test_2_1_states_the_mass_prior_it_was_given_not_a_constant(tmp_path):
    """The contract's ``mass_prior_kind`` is read off the file's basis, so it is
    only worth comparing PE-to-PE if the basis is what was ingested.  Every
    export used to stamp ``"uniform_detector_frame"`` regardless -- including
    the 9 shipped rows whose analytic prior was never parsed."""
    a = _export(tmp_path, "gwcat2.1", "component", "massprior21.h5")
    contract = json.loads(a["contract"])
    assert contract["mass_prior_kind"] == a["mass_prior_basis"]
    # this three-event slice IS fully parsed, so the verified claim survives
    assert a["mass_prior_basis"] == "uniform_detector_frame"
    assert bool(a["mass_prior_verified"]) is True
    assert int(a["n_events_mass_prior_unverified"]) == 0


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


@needs_store
def test_2_1_contract_states_the_selection_of_the_view_it_came_from(tmp_path):
    """The production reproduction (GW-33): exporting ``cat.select("bbh")``
    retained all 273 BBH rows while the file recorded no class filter, no cut
    estimator and no event list."""
    from gwcat.catalog import GWCatalog

    cat = GWCatalog(STORE)
    bbh = cat.select(source_class="bbh")
    out = tmp_path / "bbh_view.h5"
    with pytest.warns(UserWarning, match="POSTERIOR MEDIAN"):
        bbh.export(str(out), format="gwcat2.1", spin_basis="component",
                   nsamp=16, seed=0, cosmology=(67.74, 0.3089))
    with h5py.File(out, "r") as f:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v)
                 for k, v in f.attrs.items()}
    assert int(attrs["nobs"]) == bbh.n_events
    contract = json.loads(attrs["contract"])
    assert contract["source_class"] == "BBH"
    assert contract["event_list_digest"] == attrs["event_list_digest"]
    assert contract["selection_spec_digest"] == attrs["selection_spec_digest"]
    # and the class filter reaches the HASH, so this file no longer pairs
    # silently with an unfiltered selection function.
    unfiltered = _export(tmp_path, "gwcat2.1", "component", "unfiltered.h5")
    assert attrs["contract_hash"] != unfiltered["contract_hash"]


def test_both_2_1_writers_are_registered():
    from gwcat.export.registry import list_formats

    fmts = set(list_formats())
    assert ("gwcat2.1", "pe") in fmts
    assert ("gwcat2.1", "selection") in fmts
    # and 2.0 is still there, unchanged
    assert ("gwcat2", "pe") in fmts
    assert ("gwcat2", "selection") in fmts


# --------------------------------------------------------------------------
# 3. chieff_chip is demoted to opt-in (GW-23)
# --------------------------------------------------------------------------
@needs_store
def test_chieff_chip_refuses_without_opting_in(tmp_path):
    """It is a PROJECTION, so R1 makes it undefinable against the O4 campaigns
    at all, and it is where the support contract bites hardest (41 of 282 events
    out of support; GW150914's reweighting collapsed to ESS = 1.0 of 3337 under
    the old floor).  The message has to say all of that, and point at component.
    """
    from gwcat.export.pe_builder import ProjectionBasisNotAllowed

    with pytest.raises(ProjectionBasisNotAllowed) as exc:
        _export(tmp_path, "gwcat2", "chieff_chip", "cc.h5")
    msg = str(exc.value)
    assert "opt-in" in msg
    assert "component" in msg            # names the remedy
    assert "R1" in msg or "uniform" in msg
    assert "allow_projection_basis" in msg


@needs_store
def test_chieff_chip_still_works_when_opted_in(tmp_path):
    """Demoted, not removed -- an analyst who needs the projected density and
    understands why can still have it."""
    from gwcat.catalog import GWCatalog

    cat = GWCatalog(STORE)
    names = list(cat.event_names[:3])
    out = tmp_path / "cc_ok.h5"
    cat.export(str(out), format="gwcat2", spin_basis="chieff_chip",
               allow_projection_basis=True, nsamp=64, seed=0,
               cosmology=(67.74, 0.3089), allowed_names=names,
               allowed_names_authoritative=True,
               allow_out_of_support=True)
    with h5py.File(out, "r") as f:
        assert "chip" in f
        v = f.attrs["spin_basis"]
        assert (v.decode() if isinstance(v, bytes) else v) == "chieff_chip"


@needs_store
def test_the_other_bases_are_unaffected_by_the_gate(tmp_path):
    for basis in ("chieff", "component", "nospin"):
        a = _export(tmp_path, "gwcat2", basis, f"un_{basis}.h5")
        assert a["format_version"] == "gwcat-pe-2.0"


def test_projection_cost_is_advertised_on_the_block():
    """So a CLI can warn before someone spends a minute per 1e6 points."""
    from gwcat.params import BLOCKS

    assert BLOCKS["spin.chieff_chip"].estimated_cost_per_point > 0
    assert BLOCKS["spin.component_polar"].estimated_cost_per_point == 0.0
