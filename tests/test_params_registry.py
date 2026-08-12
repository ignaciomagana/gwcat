"""Tests for the ParameterBlock registry (GW-17, design PR-A).

GW-17 is pure assembly -- nothing in gwcat imports ``gwcat.params`` yet -- so
these tests pin the *contract* rather than any exported file:

  * the registry's store requirements agree with ``schema.EXPORT_REQUIREMENTS``
    for the three legacy spin bases, so wiring the builders (GW-18/19) cannot
    silently change what an export demands;
  * every block round-trips ``describe()`` into JSON-able provenance;
  * the declarative physics invariants hold (R1: a projection must constrain its
    parent; a column cannot be both fittable and advisory);
  * the projection-vs-bijection distinction actually separates the bases the way
    the measurements say it does.
"""
import json

import numpy as np
import pytest

from gwcat import schema
from gwcat.params import (BLOCKS, LEGACY_SPIN_BASES, SPACES,
                          CampaignRequirement, ParameterBlock, Range,
                          get_space, list_spaces)


# --------------------------------------------------------------------------
# 1. The registry agrees with the existing schema contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize("basis", LEGACY_SPIN_BASES)
def test_store_required_matches_schema_export_requirements(basis):
    """Wiring the builders must not change what an export demands."""
    expected = set(schema.EXPORT_REQUIREMENTS[f"gwcat2_pe:{basis}"])
    assert set(get_space(basis).store_required) == expected


def test_component_alternatives_match_the_schema_tilt_groups():
    alts = set(get_space("component").store_alternatives)
    for group in schema.COMPONENT_TILT_ALTERNATIVES:
        assert group in alts


def test_chieff_chip_alternatives_match_the_schema_chip_group():
    alts = set(get_space("chieff_chip").store_alternatives)
    for group in schema.CHIEFF_CHIP_CHIP_ALTERNATIVES:
        assert group in alts


# --------------------------------------------------------------------------
# 2. Provenance round-trips
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(BLOCKS))
def test_every_block_describes_itself_as_json(name):
    d = BLOCKS[name].describe()
    json.loads(json.dumps(d))              # must be JSON-able for the attrs
    assert d["name"] == name
    assert d["kind"] in ("mass", "distance", "sky", "spin", "tidal")
    assert d["map_kind"] in ("bijective", "projection")


@pytest.mark.parametrize("name", sorted(SPACES))
def test_every_space_describes_itself_as_json(name):
    d = get_space(name).describe()
    json.loads(json.dumps(d))
    assert d["space"] == name
    assert d["blocks"]
    # the block provenance is carried, not just the names
    assert len(d["block_provenance"]) == len(d["blocks"])


# --------------------------------------------------------------------------
# 3. Declarative invariants (R1 and the fit/advisory split)
# --------------------------------------------------------------------------
def test_a_projection_claiming_exactness_must_constrain_its_parent():
    """R1 as an invariant, not a convention.

    Converting a component draw density into a projected one integrates out
    degrees of freedom at fixed coordinate, which has a closed form only for a
    constrained parent.  A block that claims otherwise is rejected at
    construction.
    """
    with pytest.raises(ValueError, match="R1"):
        ParameterBlock(name="spin.bogus", kind="spin", columns=("x",),
                       map_kind="projection", exact_draw_density=True)
    # ... and it is fine once the requirement is declared
    ok = ParameterBlock(name="spin.ok", kind="spin", columns=("x",),
                        map_kind="projection", exact_draw_density=True,
                        requires_campaign=CampaignRequirement(
                            uniform_magnitude=True))
    assert ok.map_kind == "projection"


def test_a_column_cannot_be_both_fittable_and_advisory():
    with pytest.raises(ValueError, match="both"):
        ParameterBlock(name="spin.dup", kind="spin", columns=("chieff",),
                       advisory_columns=("chieff",))


def test_ranges_must_refer_to_real_columns():
    with pytest.raises(ValueError, match="non-existent"):
        ParameterBlock(name="spin.r", kind="spin", columns=("a1",),
                       ranges={"nope": Range(0.0, 1.0)})


def test_unknown_kind_and_map_kind_rejected():
    with pytest.raises(ValueError, match="kind must be"):
        ParameterBlock(name="x", kind="colour")
    with pytest.raises(ValueError, match="map_kind"):
        ParameterBlock(name="x", kind="spin", map_kind="sideways")


def test_two_blocks_cannot_claim_the_same_fit_column():
    from gwcat.params.registry import ParameterSpace

    a = ParameterBlock(name="spin.a", kind="spin", columns=("chieff",))
    b = ParameterBlock(name="spin.b", kind="spin", columns=("chieff",))
    with pytest.raises(ValueError, match="claimed as a FIT column by both"):
        ParameterSpace("bad", (a, b))


# --------------------------------------------------------------------------
# 4. The distinction the whole design rests on
# --------------------------------------------------------------------------
def test_component_is_exact_and_chieff_is_not():
    """The load-bearing claim: the component basis assumes nothing about the
    campaign, the projections assume a uniform-isotropic parent."""
    comp = get_space("component")
    assert comp.is_exact is True
    assert comp.projections == ()
    assert comp.spin_block.requires_campaign.any_required is False

    for basis in ("chieff", "chieff_chip"):
        sp = get_space(basis)
        assert sp.is_exact is False
        assert len(sp.projections) == 1
        req = sp.spin_block.requires_campaign
        assert req.uniform_magnitude and req.isotropic_tilts and req.single_amax


def test_flat_box_spin_priors_are_per_event_constant():
    """Why the component basis is both amax-robust and weight-variance-free:
    a constant factor cancels in the consumer's per-event p_pe normalisation.
    Measured: ESS/nsamp median 0.861 (component) vs 0.589 (chieff)."""
    assert BLOCKS["spin.component_polar"].prior_is_per_event_constant is True
    assert BLOCKS["spin.component_6d"].prior_is_per_event_constant is True
    # cartesian is the exception -- its a^-2 factor is per-SAMPLE
    assert BLOCKS["spin.cartesian"].prior_is_per_event_constant is False
    assert BLOCKS["spin.chieff"].prior_is_per_event_constant is False


def test_chieff_and_chip_are_advisory_in_the_component_space():
    """The live hole sel_v2_component.h5 shipped with: it carries a `chip`
    dataset whose draw density is not in `pdraw`, and nothing said so."""
    comp = get_space("component")
    assert "chip" in comp.advisory_columns
    assert "chieff" in comp.advisory_columns
    assert "chip" not in comp.fit_columns
    assert "chieff" not in comp.fit_columns
    # in the space that OWNS them they are fittable
    assert "chip" in get_space("chieff_chip").fit_columns
    assert "chieff" in get_space("chieff").fit_columns


def test_sky_is_in_every_space_unconditionally():
    for name in list_spaces():
        assert "ra" in get_space(name).fit_columns
        assert "dec" in get_space(name).fit_columns


def test_core_blocks_are_in_every_space():
    for name in list_spaces():
        kinds = {b.kind for b in get_space(name).blocks}
        assert {"mass", "distance", "sky", "spin"} <= kinds


# --------------------------------------------------------------------------
# 5. rng neutrality (the gate that protects chieff byte-parity)
# --------------------------------------------------------------------------
def test_chieff_space_fetches_no_extra_store_columns():
    """The chieff byte-parity contract holds only because that path fetches no
    extra per-event columns and therefore never perturbs the default_rng
    stream.  Encoded here so GW-18 cannot break it silently."""
    assert get_space("chieff").store_params_fetched == ()
    assert get_space("nospin").store_params_fetched == ()
    # the non-chieff spaces DO fetch extras, which is why they are exempt
    assert get_space("component").store_params_fetched


# --------------------------------------------------------------------------
# 6. Campaign requirements against real spin_meta shapes
# --------------------------------------------------------------------------
def test_unmet_requirements_flags_a_measured_non_uniform_campaign():
    """The O4ab shape: checks RAN and FAILED -> a real contradiction."""
    meta = {"uniform_isotropic": False,
            "amax_detected": (0.998, 0.998),
            "checks": {"magnitude_uniform": (False, False),
                       "isotropy_dev": (0.6419, 0.6419)}}
    unmet = get_space("chieff").unmet_requirements(meta)
    assert "spin.chieff" in unmet
    assert "uniform_magnitude" in unmet["spin.chieff"]
    # the component space requires nothing, so it is unaffected
    assert get_space("component").unmet_requirements(meta) == {}


def test_unverifiable_campaign_is_not_reported_as_a_violation():
    """A file carrying no spin draw densities is unknown, not contradicted --
    conflating the two would refuse every legacy spin-less campaign (GW-06)."""
    meta = {"uniform_isotropic": False, "amax_detected": (0.99, 0.99),
            "checks": {}}
    assert get_space("chieff").unmet_requirements(meta) == {}


def test_undetectable_amax_is_a_violation_for_a_projection():
    """GW-05: a non-uniform draw yields amax=None rather than a fabricated
    number, and a projection needs a real one."""
    meta = {"uniform_isotropic": True, "amax_detected": (None, None),
            "checks": {"magnitude_uniform": (True, True)}}
    unmet = get_space("chieff_chip").unmet_requirements(meta)
    assert "single_amax" in unmet["spin.chieff_chip"]


# --------------------------------------------------------------------------
# 7. Range semantics
# --------------------------------------------------------------------------
def test_range_closedness():
    r = Range(0.0, 2.0 * np.pi, closed="left")
    assert bool(r.contains(0.0)) is True
    assert bool(r.contains(2.0 * np.pi)) is False
    both = Range(-1.0, 1.0)
    np.testing.assert_array_equal(both.contains(np.array([-1.0, 0.0, 1.0, 1.1])),
                                  np.array([True, True, True, False]))


def test_sky_ranges_are_radians_not_degrees_or_colatitude():
    """A degrees or colatitude ingest passes every check today; the declared
    range is what makes it catchable."""
    sky = BLOCKS["sky.radec"]
    assert sky.ranges["dec"].lo == pytest.approx(-np.pi / 2)
    assert sky.ranges["dec"].hi == pytest.approx(np.pi / 2)
    assert bool(sky.ranges["dec"].contains(1.0)) is True     # radians
    assert bool(sky.ranges["dec"].contains(45.0)) is False   # degrees
    assert bool(sky.ranges["dec"].contains(2.0)) is False    # colatitude


# --------------------------------------------------------------------------
# 8. The densities that are wired already
# --------------------------------------------------------------------------
def test_component_polar_prior_is_the_flat_box_constant():
    from gwcat.params import PEContext

    ctx = PEContext(amax_1=0.99, amax_2=0.5)
    lnp = BLOCKS["spin.component_polar"].ln_prior_pe({}, ctx)
    assert lnp == pytest.approx(-np.log(4.0 * 0.99 * 0.5))
    assert np.ndim(lnp) == 0, "must be a scalar: it is constant within an event"


def test_mass_block_jacobian_and_its_gate():
    from gwcat.params import PEContext

    cols = {"m1det": np.array([10.0, 40.0])}
    ok = BLOCKS["mass.det_pair"].ln_prior_pe(
        cols, PEContext(mass_prior_kind="uniform_detector_frame"))
    np.testing.assert_allclose(ok, np.log(cols["m1det"]))
    # a prior that is not flat in the components invalidates the Jacobian
    with pytest.raises(ValueError, match="Jacobian"):
        BLOCKS["mass.det_pair"].ln_prior_pe(
            cols, PEContext(mass_prior_kind="uniform_chirp_mass"))


def test_sky_measure_is_declared_on_both_sides():
    """It cancels only if BOTH sides carry it -- previously an assumption
    nobody had written down (the review's one unresolved finding)."""
    sky = BLOCKS["sky.radec"]
    assert sky.ln_prior_pe({}, None) == pytest.approx(-np.log(4 * np.pi))
    assert sky.ln_draw_inj({}, None) == pytest.approx(-np.log(4 * np.pi))
    assert sky.ln_prior_pe({}, None) == sky.ln_draw_inj({}, None)


def test_component_support_predicate_matches_the_builder_rule():
    from gwcat.params import PEContext

    cols = {"a1": np.array([0.5, 0.995]), "a2": np.array([0.1, 0.2]),
            "cost1": np.array([0.0, 0.0]), "cost2": np.array([0.0, 0.0])}
    sup = BLOCKS["spin.component_polar"].support(
        cols, PEContext(amax_1=0.99, amax_2=0.99))
    np.testing.assert_array_equal(sup, np.array([True, False]))


def test_mass_and_distance_contribute_nothing_on_the_injection_side():
    """They live in pdraw_base, which is deliberately not a block."""
    assert BLOCKS["mass.det_pair"].ln_draw_inj({}, None) == 0.0
    assert BLOCKS["distance.dl"].ln_draw_inj({}, None) == 0.0


def test_the_pe_builder_now_drives_itself_from_the_registry():
    """GW-17 asserted that nothing imported ``gwcat.params``; GW-18 wired the PE
    builder to it, so that tripwire is replaced by its inverse.

    The point is not merely that the import exists -- it is that the builder no
    longer carries its own copy of the per-basis requirement ladder.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "gwcat" / "export" / "pe_builder.py").read_text()
    assert "from ..params import get_space" in src
    assert "space = get_space(spin_basis)" in src
    # the three-branch requirement ladder is gone
    assert 'need = list(COMPONENT_REQUIRED)' not in src
    assert 'need = list(DARKSIRENS_REQUIRED)' not in src
    # and the extras gate is derived, not asserted
    assert 'need_extras = spin_basis != "chieff"' not in src
