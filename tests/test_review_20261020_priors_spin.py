"""Review 2026-10-20, priors and spin package: regression tests.

Each test is the review's failure scenario, reduced to a synthetic fixture.
The production exports (BBH chieff PE, O3+O4 chieff_reference selection) are
byte-identical in their datasets; these tests pin the behaviour around them.
"""
import json
import warnings

import h5py
import numpy as np
import pytest

from gwcat.catalog import GWCatalog
from gwcat.export.validate import _check_pe_prior_impl, validate_export_v2
from gwcat.release_cosmology import (LEGACY_TABLE_FILENAME, bundled_table_path,
                                     load_release_cosmology_table)

from test_export_v2_pe import _build_spin_store

_COSMO = (67.74, 0.3089)


def _collect():
    fails, results = [], {}

    def _fail(name, msg):
        fails.append(name)
        raise ValueError(f"{name}: {msg}")
    return fails, results, _fail


def _add_meta(store, **cols):
    with h5py.File(store, "r+") as f:
        for k, v in cols.items():
            if k in f["meta"]:
                del f["meta"][k]
            if isinstance(v[0], str):
                f["meta"].create_dataset(k, data=np.array(
                    v, dtype=h5py.string_dtype()))
            else:
                f["meta"].create_dataset(k, data=np.asarray(v, dtype=float))


def _export(store, out, **kw):
    kw = dict(dict(format="gwcat2", spin_basis="chieff", nsamp=64, seed=0,
                   cosmology=_COSMO), **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(store).export(str(out), **kw)
    with h5py.File(out, "r") as f:
        return dict(f.attrs)


def _strs(v):
    return [x.decode() if isinstance(x, bytes) else str(x)
            for x in np.atleast_1d(v)]


# ==========================================================================
# The release-reweight table's identity reaches the PE product
# ==========================================================================
def _two_event_store(tmp_path, name="store.h5"):
    events = [{"name": "GW190521_030229"}, {"name": "GW230708_053705"}]
    store, _ = _build_spin_store(tmp_path, events, n_per_event=200, name=name)
    return store


def test_pe_product_records_the_release_reweight_table(tmp_path):
    """The failure scenario: a build with the legacy (astropy) table must be
    distinguishable from the file, and --require-exact-priors must refuse it.
    """
    legacy = load_release_cosmology_table(
        bundled_table_path(LEGACY_TABLE_FILENAME))
    store = _two_event_store(tmp_path)
    _add_meta(store,
              dL_prior_cosmology_source=["legacy_gwcat_default",
                                         "analytic_declared"],
              dL_prior_cosmology_table_sha256=[legacy.sha256, ""],
              dL_prior_impl=["exact", "exact"])
    out = tmp_path / "pe.h5"
    attrs = _export(store, out)
    assert _strs(attrs["dL_prior_cosmology_source_per_event"]) == [
        "legacy_gwcat_default", "analytic_declared"]
    assert _strs(attrs["dL_prior_cosmology_table_sha256_per_event"]) == [
        legacy.sha256, ""]
    ids = json.loads(attrs["release_reweight_cosmology_table_ids"])
    assert ids == {legacy.sha256: legacy.table_id}

    fails, results, _fail = _collect()
    with pytest.raises(ValueError, match="regression-only release-reweight"):
        _check_pe_prior_impl(_fail, results, attrs, require_exact=True)
    with pytest.raises(ValueError, match="pe_prior_impl_exact"):
        validate_export_v2(str(out), require_exact_priors=True)


def test_production_table_passes_require_exact(tmp_path):
    prod = load_release_cosmology_table()
    store = _two_event_store(tmp_path)
    _add_meta(store,
              dL_prior_cosmology_source=["documented", "analytic_declared"],
              dL_prior_cosmology_table_sha256=[prod.sha256, ""],
              dL_prior_impl=["exact", "exact"])
    out = tmp_path / "pe.h5"
    attrs = _export(store, out)
    assert json.loads(attrs["release_reweight_cosmology_table_ids"]) == {
        prod.sha256: prod.table_id}
    r = validate_export_v2(str(out), require_exact_priors=True)
    assert r["pe_prior_impl_exact"] is True


def test_a_file_without_the_record_only_warns():
    attrs = {"spin_basis": "chieff", "chi_eff_prior_impl": "exact",
             "event_names": np.array(["GW1"], dtype=object),
             "dL_prior_impl_per_event": np.array(["exact"], dtype=object)}
    fails, results, _fail = _collect()
    with pytest.warns(UserWarning, match="cannot be verified"):
        _check_pe_prior_impl(_fail, results, attrs, require_exact=True)
    assert not fails


# ==========================================================================
# The ingest-time selection reason and the mass prior source reach the product
# ==========================================================================
def test_pe_product_records_the_ingest_selection_reason(tmp_path):
    """Priority-list picks and last-resort picks must not read the same."""
    store = _two_event_store(tmp_path)
    _add_meta(store,
              selection_reason=["preferred_priority", "preferred_last_resort"],
              priority_rank=[0.0, 3.0],
              mass_prior_source=["analytic[C01:IMRPhenomXPHM]",
                                 "constituent_mixture:own_prior[C00:A+C00:B]"])
    attrs = _export(store, tmp_path / "pe.h5")
    assert _strs(attrs["sample_set_ingest_selection_reason_per_event"]) == [
        "preferred_priority", "preferred_last_resort"]
    np.testing.assert_array_equal(attrs["sample_set_priority_rank_per_event"],
                                  [0.0, 3.0])
    assert _strs(attrs["mass_prior_source_per_event"]) == [
        "analytic[C01:IMRPhenomXPHM]",
        "constituent_mixture:own_prior[C00:A+C00:B]"]


def test_store_without_the_columns_records_empty_provenance(tmp_path):
    store = _two_event_store(tmp_path)
    attrs = _export(store, tmp_path / "pe.h5")
    assert _strs(attrs["sample_set_ingest_selection_reason_per_event"]) == [
        "", ""]
    assert np.all(np.isnan(attrs["sample_set_priority_rank_per_event"]))
    assert json.loads(attrs["release_reweight_cosmology_table_ids"]) == {}


# ==========================================================================
# A NaN coordinate is an error, not an out-of-support sample
# ==========================================================================
def _store_with_nan(tmp_path, column, name="nan_store.h5"):
    """Two events; event 2 carries one NaN sample in ``column`` although the
    availability mask marks the column present."""
    store = _two_event_store(tmp_path, name=name)
    with h5py.File(store, "r+") as f:
        offs = f["index/offsets"][()] if "offsets" in f["index"] else None
        col = f[f"samples/{column}"]
        arr = col[()]
        start = int(offs[1]) if offs is not None else arr.size // 2
        arr[start + 3] = np.nan
        col[...] = arr
    return store


def test_in_support_helper_refuses_a_nan_chi_eff():
    from gwcat.spin import (NonFiniteSpinInputError,
                            chi_eff_prior_logprob_in_support)
    chi = np.array([np.nan, 0.1])
    m = np.array([30.0, 30.0]), np.array([20.0, 20.0])
    with pytest.raises(NonFiniteSpinInputError, match="non-finite"):
        chi_eff_prior_logprob_in_support(chi, *m, amax=0.99)
    with pytest.raises(NonFiniteSpinInputError, match="m2_source"):
        chi_eff_prior_logprob_in_support(np.array([0.1, 0.1]), m[0],
                                         np.array([20.0, np.nan]), amax=0.99)


def test_in_support_helper_keeps_finite_semantics():
    from gwcat.spin import chi_eff_prior_logprob_in_support
    lp, sup = chi_eff_prior_logprob_in_support(
        np.array([0.1, 0.995, -0.999]), np.array([50.0] * 3),
        np.array([25.0] * 3), amax=0.99)
    assert sup.tolist() == [True, False, False]
    assert np.isfinite(lp[0]) and np.all(lp[1:] == -np.inf)


@pytest.mark.parametrize("column", ["chi_eff", "mass_2"])
def test_pe_export_names_the_nan_column_and_event(tmp_path, column):
    """The failure scenario: a NaN chi_eff (or mass) was reported as 'the
    declared amax does not cover the samples'.  It now names the column and
    the event."""
    from gwcat.schema import NonFiniteSampleError
    store = _store_with_nan(tmp_path, column)
    with pytest.raises(NonFiniteSampleError, match="GW230708_053705: 1"):
        _export(store, tmp_path / "pe.h5", nsamp=200)   # every sample drawn


def test_nospin_export_refuses_a_nan_mass(tmp_path):
    """Review scenario: nospin with a NaN mass_2 was written and passed
    validate --strict."""
    from gwcat.schema import NonFiniteSampleError
    store = _store_with_nan(tmp_path, "mass_2")
    with pytest.raises(NonFiniteSampleError, match="'m2det'"):
        _export(store, tmp_path / "pe.h5", spin_basis="nospin", nsamp=200)


# ==========================================================================
# The chi_eff support is the true, mass-dependent edge
# ==========================================================================
def test_grid_density_past_the_true_edge_is_masked():
    """amax = (0.99, 0.05): the true edge is (m1 0.99 + m2 0.05)/(m1+m2).
    The legacy grid interpolates positive density past it between its q rows;
    the in-support helper now masks it to zero."""
    from gwcat.spin import chi_eff_prior_logprob_in_support, get_chi_eff_prior
    prior = get_chi_eff_prior(0.99, 0.05, "grid")
    qq = 0.5 * (prior.q_grid[60] + prior.q_grid[61])
    m1 = np.full(50, qq * 50.0)
    m2 = np.full(50, (1.0 - qq) * 50.0)
    edge = (m1 * 0.99 + m2 * 0.05) / (m1 + m2)
    chi = edge + np.linspace(1e-5, 0.01, 50)
    assert np.any(prior.prob(chi, m1, m2) > 0)       # the leak itself
    assert np.all(prior.support(chi))                 # the necessary bound
    assert not np.any(prior.support(chi, m1, m2))     # the true edge
    lp, sup = chi_eff_prior_logprob_in_support(chi, m1, m2, amax=0.99,
                                               amax_2=0.05, impl="grid")
    assert not sup.any() and np.all(lp == -np.inf)
    lp_x, sup_x = chi_eff_prior_logprob_in_support(chi, m1, m2, amax=0.99,
                                                   amax_2=0.05, impl="exact")
    assert not sup_x.any()


# ==========================================================================
# Spin ceilings must be spin ceilings
# ==========================================================================
@pytest.mark.parametrize("bad", ["nan", float("nan"), "inf", -0.1, 0.0, 1.7])
def test_parse_amax_option_refuses_non_ceilings(bad):
    from gwcat.spin import parse_amax_option
    with pytest.raises(ValueError, match="spin ceiling"):
        parse_amax_option(bad)


@pytest.mark.parametrize("good,want", [("auto", None), (None, None),
                                       ("0.99", 0.99), (1.0, 1.0)])
def test_parse_amax_option_accepts_ceilings(good, want):
    from gwcat.spin import parse_amax_option
    assert parse_amax_option(good) == want


def test_nan_amax_export_is_refused(tmp_path):
    """Review scenario: --amax nan wrote p_pe times leftover memory, and the
    file validated."""
    store = _two_event_store(tmp_path)
    with pytest.raises(ValueError, match="spin ceiling"):
        _export(store, tmp_path / "pe.h5", amax="nan")
    with pytest.raises(ValueError, match="amax_fallback"):
        _export(store, tmp_path / "pe2.h5", amax_fallback=float("nan"))


# ==========================================================================
# Event map: duplicates, the substitute flag, and the substitute chain
# ==========================================================================
from test_waveform_event_map import (E1, E2, E3, NR, XP,  # noqa: E402
                                     _store as _map_store)


def _map_export(store, out, **kw):
    kw = dict(dict(format="gwcat2", spin_basis="chieff", nsamp=32, seed=0,
                   cosmology=_COSMO), **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWCatalog(store).export(str(out), **kw)
    with h5py.File(out, "r") as f:
        return dict(f.attrs)


def _two_release_store(tmp_path, n=400):
    """E1 twice: release A (XPHM-ST with no mass below q = 1/6, NRSur) and
    release B (XPHM-ST with 50% below q = 1/6, NRSur)."""
    events = [{"name": E1, "waveform": wf, "n": n}
              for wf in (XP, NR, XP, NR)]
    store, _ = _build_spin_store(tmp_path, events, name="two_releases.h5")
    with h5py.File(store, "r+") as f:
        off = f["index/offsets"][:]
        m1 = f["samples/mass_1"][:]
        m2 = f["samples/mass_2"][:]
        m2[off[0]:off[1]] = np.maximum(m2[off[0]:off[1]], 0.5 * m1[off[0]:off[1]])
        lo = int(off[2])
        m2[lo:lo + n // 2] = 0.1 * m1[lo:lo + n // 2]
        f["samples/mass_2"][...] = m2
    return store


def test_event_map_refuses_duplicate_event_label_rows(tmp_path):
    """Review scenario: the 1% rule passed on release A's 0% and the export
    took release B's NRSur samples, with no warning."""
    from gwcat.waveform_policy import SampleSetMapError
    store = _two_release_store(tmp_path)
    with pytest.raises(SampleSetMapError, match="more than one store row"):
        _map_export(store, tmp_path / "pe.h5", waveform_policy="event-map",
                    sample_set_map={E1: f"C01:{NR}"}, nrsur_q_rule=0.01)
    with pytest.raises(SampleSetMapError, match="more than one store row"):
        _map_export(store, tmp_path / "pe2.h5", waveform_policy="event-map",
                    sample_set_map={E1: f"C01:{NR}"})


def test_q_rule_substitute_needs_the_rule_and_the_event_map(tmp_path):
    store = _map_store(tmp_path)
    with pytest.raises(ValueError, match="needs nrsur_q_rule"):
        _map_export(store, tmp_path / "pe.h5", waveform_policy="event-map",
                    sample_set_map={E1: f"C01:{NR}", E2: f"C01:{NR}",
                                    E3: f"C01:{XP}"},
                    nrsur_q_rule_substitute=True)
    with pytest.raises(ValueError, match="apply only to"):
        _map_export(store, tmp_path / "pe2.h5", waveform_policy="mixed-first",
                    nrsur_q_rule_substitute=True)


def test_auto_substitution_keeps_the_declared_substitute_chain(tmp_path):
    """The rule replaces a DECLARED substitute: the map's own label and the
    declared label and reason stay in the product."""
    store = _map_store(tmp_path, q_frac={E1: 0.5})
    smap = {"events": {E1: "C01:SEOBNRv5PHM", E2: f"C01:{NR}",
                       E3: f"C01:{XP}"},
            "substitutes": {E1: {"label": f"C01:{NR}",
                                 "reason": "seob_not_released"}}}
    attrs = _map_export(store, tmp_path / "pe.h5", waveform_policy="event-map",
                        sample_set_map=smap, nrsur_q_rule=0.01,
                        nrsur_q_rule_substitute=True)
    assert _strs(attrs["sample_set_substitute_events"]) == [E1]
    assert _strs(attrs["sample_set_substitute_labels"]) == [f"C01:{XP}"]
    assert _strs(attrs["sample_set_substitute_original_labels"]) == [
        "C01:SEOBNRv5PHM"]
    assert _strs(attrs["sample_set_substitute_reasons"]) == [
        f"nrsur_q_rule;replaced=C01:{NR}(seob_not_released)"]


def test_json_map_with_a_duplicated_event_is_refused(tmp_path):
    from gwcat.waveform_policy import SampleSetMapError, load_sample_set_map
    p = tmp_path / "map.json"
    p.write_text('{"events": {"%s": "C01:%s", "%s": "C01:%s"}}'
                 % (E1, NR, E1, XP))
    with pytest.raises(SampleSetMapError, match="duplicate key"):
        load_sample_set_map(str(p))


# ==========================================================================
# Registry: campaign requirements are evaluated, not only declared
# ==========================================================================
def _o4_with_azimuth(tmp_path, uniform):
    from test_selection_spin import write_o4_full
    path = write_o4_full(tmp_path / ("o4_uni.hdf" if uniform else "o4_az.hdf"),
                         400, seed=3)
    if not uniform:
        with h5py.File(path, "r+") as f:
            ph = f["events/spin1_azimuthal_angle"][()]
            # a normalised but NON-uniform azimuth density, 1 + 0.5 cos(phi)
            f["events/lnpdraw_spin1_azimuthal_angle"][...] = (
                np.log1p(0.5 * np.cos(ph)) - np.log(2 * np.pi))
    return path


def test_factored_campaign_azimuths_are_marginalised_not_refused(tmp_path):
    """A factored campaign gives each azimuth density as its own factor, so
    the 4-D component density is exact whatever that factor is: a normalised
    NON-uniform azimuth density must leave pdraw unchanged, and the product
    records how the azimuths were treated."""
    from gwcat.export import build_selection_product
    from gwcat.selection import SelectionSet
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uni = build_selection_product(
            SelectionSet(_o4_with_azimuth(tmp_path, True)),
            spin_basis="component")
        non = build_selection_product(
            SelectionSet(_o4_with_azimuth(tmp_path, False)),
            spin_basis="component")
    np.testing.assert_array_equal(non.columns["pdraw"], uni.columns["pdraw"])
    assert _strs(non.attrs["injected_spin_azimuths_per_campaign"]) == [
        "factored"]


def test_joint_density_campaigns_record_the_azimuth_assumption(tmp_path):
    """The review case: joint_polar, joint_cartesian and the O3 cartesian
    format reduce a joint density to 4-D by dividing the azimuths out as
    uniform, which nothing checked or recorded.  The O3 format's per-spin
    isotropy check verifies it; a joint density cannot, so it is recorded as
    an assumption."""
    from gwcat.export import build_selection_product
    from gwcat.selection import SelectionSet
    from test_selection_spin import write_endo3_full, write_mixture
    cases = {"assumed_uniform": [write_mixture(tmp_path / "p.hdf", "polar"),
                                 write_mixture(tmp_path / "c.hdf",
                                               "cartesian")],
             "checked_isotropic": [write_endo3_full(tmp_path / "e.hdf", 200)]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for want, paths in cases.items():
            for path in paths:
                for basis, kw in (("component", {}),
                                  ("chieff_reference",
                                   {"spin_reference_amax": 0.99})):
                    prod = build_selection_product(SelectionSet(str(path)),
                                                   spin_basis=basis, **kw)
                    assert _strs(prod.attrs[
                        "injected_spin_azimuths_per_campaign"]) == [want], (
                        path, basis)


def test_only_a_block_that_keeps_the_azimuths_needs_them_uniform():
    """A failed azimuth check contradicts the 6-D block, which keeps the
    azimuths at an assumed uniform density; the 4-D component block
    marginalises a factored azimuth density exactly and requires nothing."""
    from gwcat.params import get_space
    meta = {"uniform_isotropic": True, "amax_detected": (0.9, 0.7),
            "checks": {"azimuth_uniform": (False, True)}}
    assert get_space("component_6d").unmet_requirements(meta) == {
        "spin.component_6d": ("uniform_azimuth",)}
    assert get_space("component").unmet_requirements(meta) == {}
    assert get_space("chieff_reference").unmet_requirements(meta) == {}


def test_21_selection_with_no_sky_in_any_row_is_refused(tmp_path):
    """Review scenario: a 2.1 selection was written and validated with ra/dec
    as fit columns and every row NaN."""
    from gwcat.params.registry import CampaignRequirementError
    from gwcat.selection import SelectionSet
    from test_selection_cumulative_mixture import (_ANCHORS,
                                                   write_cumulative_mixture)
    path, _, _ = write_cumulative_mixture(tmp_path / "mix.hdf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(CampaignRequirementError, match="no finite row"):
            SelectionSet(path).export(str(tmp_path / "x21.h5"),
                                      format="gwcat2.1",
                                      spin_basis="component",
                                      snr_threshold=10.0,
                                      mixture_anchors=_ANCHORS)
        # 2.0 keeps writing the NaN-sky rows, as before (GW-39).
        SelectionSet(path).export(str(tmp_path / "x20.h5"), format="gwcat2",
                                  spin_basis="component", snr_threshold=10.0,
                                  mixture_anchors=_ANCHORS)


# ==========================================================================
# Spin blocks: the declared PE densities are the builders' densities
# ==========================================================================
def _ctx(amax_1, amax_2, m1, m2, name="GWx"):
    from gwcat.params import PEContext
    return PEContext(event_name=name, m1src=m1, m2src=m2, amax_1=amax_1,
                     amax_2=amax_2)


def test_chieff_block_density_uses_both_ceilings_and_the_support():
    from gwcat.params import BLOCKS
    from gwcat.spin import chi_eff_prior_logprob_in_support
    chi = np.array([0.1, 0.3, 0.7])
    m1, m2 = np.array([30.0] * 3), np.array([20.0] * 3)
    got = BLOCKS["spin.chieff"].ln_prior_pe({"chieff": chi},
                                            _ctx(0.99, 0.05, m1, m2))
    want, _ = chi_eff_prior_logprob_in_support(chi, m1, m2, amax=0.99,
                                               amax_2=0.05)
    np.testing.assert_array_equal(got, want)
    assert got[2] == -np.inf               # past (30*.99 + 20*.05)/50 = 0.614
    # and the reference block's PE half is the same density
    np.testing.assert_array_equal(
        BLOCKS["spin.chieff_reference"].ln_prior_pe(
            {"chieff": chi}, _ctx(0.99, 0.05, m1, m2)), want)


def test_chieff_chip_block_refuses_two_ceilings():
    from gwcat.params import BLOCKS
    m1, m2 = np.array([30.0]), np.array([20.0])
    with pytest.raises(ValueError, match="one shared ceiling"):
        BLOCKS["spin.chieff_chip"].ln_prior_pe(
            {"chieff": np.array([0.1]), "chip": np.array([0.3])},
            _ctx(0.99, 0.05, m1, m2))


# ==========================================================================
# Every written dataset is a declared fit or advisory column
# ==========================================================================
_WEIGHTS = {"p_pe", "pdraw", "in_support"}


def _declared(space_name):
    from gwcat.params import get_space
    sp = get_space(space_name)
    return set(sp.fit_columns) | set(sp.advisory_columns) | _WEIGHTS


@pytest.mark.parametrize("basis", ["chieff", "component", "nospin"])
def test_pe_datasets_are_all_declared(tmp_path, basis):
    store = _two_event_store(tmp_path)
    out = tmp_path / f"pe_{basis}.h5"
    _export(store, out, spin_basis=basis)
    with h5py.File(out, "r") as f:
        written = set(f.keys())
    assert written - _declared(basis) == set()


def test_chieff_selection_datasets_are_all_declared(tmp_path):
    from gwcat.selection import SelectionSet
    from test_selection_spin import write_o4_full
    path = write_o4_full(tmp_path / "o4.hdf", 200, amax=(0.9, 0.9), seed=4)
    out = tmp_path / "sel.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SelectionSet(path).export(str(out), spin_basis="chieff")
    with h5py.File(out, "r") as f:
        written = set(f.keys())
    assert {"a1", "chip"} <= written
    assert written - _declared("chieff") == set()


def test_writer_refuses_an_undeclared_dataset(tmp_path):
    from gwcat.export.pe_builder import build_pe_product
    from gwcat.export.writers_gwcat2 import write_pe_gwcat2
    store = _two_event_store(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prod = build_pe_product(GWCatalog(store), spin_basis="chieff",
                                nsamp=16, seed=0, cosmology=_COSMO)
    prod.columns["mystery"] = np.zeros_like(prod.columns["p_pe"])
    with pytest.raises(ValueError, match="mystery"):
        write_pe_gwcat2(prod, str(tmp_path / "x.h5"))


# ==========================================================================
# allow_zero_p_pe covers exact zeros only, and is recorded
# ==========================================================================
@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_allow_zero_does_not_write_nan_inf_or_negative(bad):
    from gwcat.schema import ZeroWeightError, check_p_pe_positive
    p = np.array([1.0, 0.0, bad, 2.0])
    with pytest.raises(ZeroWeightError, match="exact zeros only"):
        check_p_pe_positive(p, allow_zero=True)
    with pytest.warns(UserWarning, match="writing it anyway"):
        assert check_p_pe_positive(np.array([1.0, 0.0, 2.0]),
                                   allow_zero=True) == 1


def test_allowed_zero_p_pe_is_recorded_on_the_product(tmp_path):
    store = _two_event_store(tmp_path)
    with h5py.File(store, "r+") as f:
        p = f["samples/p_dL_pe"][()]
        p[:5] = 0.0
        f["samples/p_dL_pe"][...] = p
    attrs = _export(store, tmp_path / "pe.h5", nsamp=200,
                    allow_zero_p_pe=True)
    assert bool(attrs["allow_zero_p_pe"]) is True
    assert int(attrs["n_zero_p_pe_allowed"]) == 5
    clean = _export(_two_event_store(tmp_path, name="clean.h5"),
                    tmp_path / "pe2.h5")
    assert bool(clean["allow_zero_p_pe"]) is False
    assert int(clean["n_zero_p_pe_allowed"]) == 0


# ==========================================================================
# The missing-parameter hint offers only spaces the builder can build
# ==========================================================================
def test_hint_omits_opt_in_spaces_and_unsatisfiable_alternatives():
    from gwcat.schema import _supported_spaces_hint, spaces_supported_by
    # a_1/a_2/chi_eff but neither cos_tilt_i nor tilt_i: the component and
    # chieff_reference spaces' alternative groups cannot be met.
    have = ["mass_1", "mass_2", "luminosity_distance", "p_dL_pe", "ra", "dec",
            "chi_eff", "a_1", "a_2", "chi_p"]
    ok = spaces_supported_by(have)
    assert "component" not in ok and "chieff" in ok
    full = have + ["cos_tilt_1", "cos_tilt_2"]
    assert "component" in spaces_supported_by(full)
    hint = _supported_spaces_hint(full)
    assert "chieff_chip" not in hint and "component" in hint


def test_nan_amax_fallback_is_refused_on_every_exporter(tmp_path):
    from gwcat.selection import SelectionSet
    from test_selection_spin import write_o4_full
    store = _two_event_store(tmp_path)
    with pytest.raises(ValueError, match="amax_fallback"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            GWCatalog(store).to_darksirens(str(tmp_path / "v1.h5"), nsamp=16,
                                           amax_fallback=float("nan"))
    path = write_o4_full(tmp_path / "o4.hdf", 50, amax=(0.9, 0.9), seed=5)
    with pytest.raises(ValueError, match="amax_fallback"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SelectionSet(path).export(str(tmp_path / "sel.h5"),
                                      spin_basis="component",
                                      amax_fallback=float("nan"))
