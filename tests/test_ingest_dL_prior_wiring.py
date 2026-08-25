"""End-to-end wiring of the distance prior through ``build_store`` (GW-37).

The number downstream cosmology divides out of ``p_pe`` is ``p_dL_pe``, and it
is computed in exactly ONE place -- the :func:`gwcat.cosmology.dL_prior_prob`
call inside :func:`gwcat.ingest.build_store`.  Before these tests, every
``p_dL_pe`` in the suite was either written into a hand-built fixture
(``rng.uniform(0.1, 1.0, n)``) or zeroed to exercise a refusal: not one test
ever read back a ``p_dL_pe`` that ``build_store`` had actually computed.

:func:`gwcat.ingest.resolve_dL_prior` and :func:`gwcat.cosmology.dL_prior_prob`
were each unit-tested in isolation; the glue joining them was not.  That gap is
not theoretical -- replacing ``kind=res.kind`` with ``kind='Uniform'`` (a
silently valid, physically wrong prior) at the call site left the whole suite
green.  The scale of what that hides: UniformSourceFrame against PowerLaw(2)
measures KS = 0.271 on real files, a density ratio swinging by more than 3x
across a 300-4000 Mpc posterior, which is a directly biased H0.

So these tests assert the stored column against ``dL_prior_prob`` recomputed
with the expected kind/alpha as LITERALS -- never as ``res.kind``, which would
re-derive the very thing under test -- and assert the ``meta/dL_prior_*``
provenance equals the same literals.  The provenance is written from ``res``
independently of what was passed to ``dL_prior_prob``, so a mis-wired call
leaves it reading correctly while the density is wrong; pinning both to
literals is what makes them checkable against each other.

The second half covers ``IngestConfig.validate_prior``, which defaults to True
and which all 17 other ``build_store`` calls in the suite pass ``False`` to --
so the KS gate body ran in zero tests.
"""
import numpy as np
import pytest

from gwcat.cosmology import dL_prior_prob, make_cosmology
from gwcat.ingest import IngestConfig, PriorMismatchError, build_store

# The three repr shapes the real releases use, verbatim (as in
# tests/test_dL_prior_class.py, which unit-tests the parser on them).
REPR_POWERLAW = ("PowerLaw(alpha=2, minimum=10, maximum=10000, "
                 "name='luminosity_distance', latex_label='$d_L$', unit='Mpc', "
                 "boundary=None)")
REPR_USF_LAL = ("bilby.gw.prior.UniformSourceFrame(minimum=10.0, "
                "maximum=4000.0, cosmology='Planck15_LAL', "
                "name='luminosity_distance', latex_label='$d_L$', unit='Mpc', "
                "boundary=None)")

_ANALYSIS = "C01:IMRPhenomXPHM"


class _FakeData:
    """Minimal stand-in for a pesummary read() result (no config -> f_ref NaN)."""


def _samples(rng, n, dmin=300.0, dmax=800.0):
    return {
        "mass_1": rng.uniform(25, 50, n),
        "mass_2": rng.uniform(10, 25, n),
        "luminosity_distance": rng.uniform(dmin, dmax, n),
        "ra": rng.uniform(0, 2 * np.pi, n),
        "dec": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "chi_eff": rng.uniform(-0.4, 0.4, n),
    }


def _ingest(tmp_path, monkeypatch, *, dl_repr, prior_samples=None, cfg=None,
            filename="GWTC-3_GW950101_000101_cosmo.h5", n=400, seed=0):
    """Run the REAL build_store against a faked PESummary read."""
    import gwcat.ingest as ing

    rng = np.random.default_rng(seed)
    analyses = {_ANALYSIS: _samples(rng, n)}
    priors = {"analytic": {_ANALYSIS: {"luminosity_distance": dl_repr}}}
    if prior_samples is not None:
        priors["samples"] = {_ANALYSIS: {
            "luminosity_distance": np.asarray(prior_samples, dtype=float)}}

    monkeypatch.setattr(ing, "_read_event_pesummary",
                        lambda path: (_FakeData(), analyses,
                                      list(analyses), priors))
    path = tmp_path / filename
    path.write_bytes(b"")  # existence only; the reader is faked
    out = tmp_path / "store.h5"
    build_store([str(path)], str(out), event_table={},
                cfg=cfg or IngestConfig(validate_prior=False))
    return str(out)


def _read(store, column):
    import h5py
    with h5py.File(store, "r") as f:
        return np.asarray(f[f"samples/{column}"])


def _meta(store, column):
    import h5py
    with h5py.File(store, "r") as f:
        v = f[f"meta/{column}"][0]
    return v.decode() if isinstance(v, (bytes, bytearray)) else v


# --------------------------------------------------------------------------
# 1. The stored p_dL_pe IS the declared prior, recomputed from literals
# --------------------------------------------------------------------------
def test_build_store_p_dL_pe_is_the_declared_power_law(tmp_path, monkeypatch):
    """A native-flavour PowerLaw(alpha=2) release stores exactly that density.

    ``_nocosmo``, so the declared class IS the effective one and the assertion
    is against the repr the file carries.  The reweighted ``_cosmo`` case is a
    separate test below -- conflating them is the GW-02 confusion.
    """
    store = _ingest(tmp_path, monkeypatch, dl_repr=REPR_POWERLAW,
                    filename="GWTC-3_GW950101_000101_nocosmo.h5")
    dL = _read(store, "luminosity_distance")
    stored = _read(store, "p_dL_pe")

    # Literals, not res.kind/res.alpha: re-deriving them from the code under
    # test is how a wiring test comes to pass on a mis-wired call.
    expected = dL_prior_prob(dL, kind="PowerLaw", cosmology=None,
                             dmin=10.0, dmax=10000.0, alpha=2.0, impl="auto")
    np.testing.assert_allclose(stored, expected, rtol=1e-12, atol=0.0)

    assert _meta(store, "dL_prior_kind") == "PowerLaw"
    assert _meta(store, "dL_prior_sampling_kind") == "PowerLaw"
    assert float(_meta(store, "dL_prior_alpha")) == 2.0
    assert float(_meta(store, "dL_prior_sampling_alpha")) == 2.0
    assert float(_meta(store, "dL_prior_min")) == 10.0
    assert float(_meta(store, "dL_prior_max")) == 10000.0


def test_build_store_p_dL_pe_is_the_declared_uniform_source_frame(
        tmp_path, monkeypatch):
    """A UniformSourceFrame(Planck15_LAL) release stores exactly that density.

    Also pins the cosmology token EXACTLY: "Planck15_LAL" and "Planck15" differ
    by ~3% at the low-distance end and that difference does not cancel
    downstream, so a wiring that dropped the token would be a real bias.
    """
    store = _ingest(tmp_path, monkeypatch, dl_repr=REPR_USF_LAL,
                    filename="GWTC-3_GW950101_000101_nocosmo.h5")
    dL = _read(store, "luminosity_distance")
    stored = _read(store, "p_dL_pe")

    assert _meta(store, "dL_prior_cosmology_name") == "Planck15_LAL"
    cosmo = make_cosmology(float(_meta(store, "dL_prior_H0")),
                           float(_meta(store, "dL_prior_Om0")))
    expected = dL_prior_prob(dL, kind="UniformSourceFrame", cosmology=cosmo,
                             dmin=10.0, dmax=4000.0, alpha=None,
                             impl=_meta(store, "dL_prior_impl"))
    np.testing.assert_allclose(stored, expected, rtol=1e-12, atol=0.0)

    assert _meta(store, "dL_prior_kind") == "UniformSourceFrame"
    assert _meta(store, "dL_prior_sampling_kind") == "UniformSourceFrame"


def test_the_two_prior_classes_are_not_interchangeable(tmp_path, monkeypatch):
    """The mutation the suite used to miss: a wrong-but-valid prior class.

    Guards the assertions above against being vacuous -- if the two densities
    were numerically close, passing them would prove nothing about the wiring.
    """
    store = _ingest(tmp_path, monkeypatch, dl_repr=REPR_POWERLAW,
                    filename="GWTC-3_GW950101_000101_nocosmo.h5")
    dL = _read(store, "luminosity_distance")
    power_law = _read(store, "p_dL_pe")
    usf = dL_prior_prob(dL, kind="UniformSourceFrame",
                        cosmology=make_cosmology(67.74, 0.3075),
                        dmin=10.0, dmax=10000.0, alpha=None, impl="bilby")
    ratio = np.max(power_law / usf) / np.min(power_law / usf)
    assert ratio > 1.2, (
        f"the two candidate priors differ by only {ratio:.3f}x across this "
        f"posterior, so asserting one against the other proves nothing")


def test_cosmo_flavour_reweighting_reaches_the_density(tmp_path, monkeypatch):
    """A ``_cosmo`` release evaluates the EFFECTIVE class, not the declared one.

    GW-02: the posteriors of a ``_cosmo`` release have been reweighted off the
    ``priors/analytic`` distribution the file still records, so the density
    divided out must be the effective UniformSourceFrame -- and the correctness
    of the 81 GWTC-2.1/3 rows rests on that substitution actually reaching
    ``dL_prior_prob``, not merely the provenance columns.
    """
    store = _ingest(tmp_path, monkeypatch, dl_repr=REPR_POWERLAW,
                    filename="GWTC-3_GW950101_000101_cosmo.h5")
    assert _meta(store, "dL_prior_release_flavour") == "cosmo"
    assert _meta(store, "dL_prior_basis") == "release_reweighted"
    # The file DECLARES PowerLaw; the effective prior is UniformSourceFrame.
    assert _meta(store, "dL_prior_sampling_kind") == "PowerLaw"
    assert _meta(store, "dL_prior_kind") == "UniformSourceFrame"

    dL = _read(store, "luminosity_distance")
    cosmo = make_cosmology(float(_meta(store, "dL_prior_H0")),
                           float(_meta(store, "dL_prior_Om0")))
    expected = dL_prior_prob(dL, kind="UniformSourceFrame", cosmology=cosmo,
                             dmin=10.0, dmax=10000.0, alpha=None,
                             impl=_meta(store, "dL_prior_impl"))
    np.testing.assert_allclose(_read(store, "p_dL_pe"), expected,
                               rtol=1e-12, atol=0.0)

    # And it is NOT the declared PowerLaw -- the substitution is observable.
    declared = dL_prior_prob(dL, kind="PowerLaw", cosmology=None,
                             dmin=10.0, dmax=10000.0, alpha=2.0, impl="auto")
    assert not np.allclose(_read(store, "p_dL_pe"), declared, rtol=1e-6)


# --------------------------------------------------------------------------
# 2. The KS gate, which every other build_store test disables
# --------------------------------------------------------------------------
def _power_law_draws(n, alpha, dmin, dmax, seed):
    """Inverse-CDF draws from p(d) ~ d^alpha on [dmin, dmax]."""
    u = np.random.default_rng(seed).uniform(0, 1, n)
    p = alpha + 1.0
    return (dmin ** p + u * (dmax ** p - dmin ** p)) ** (1.0 / p)


def test_ks_gate_passes_on_matching_prior_samples(tmp_path, monkeypatch):
    """Prior samples drawn from the declared class raise nothing."""
    import warnings as _w
    samples = _power_law_draws(20000, 2.0, 10.0, 10000.0, seed=1)
    with _w.catch_warnings(record=True) as record:
        _w.simplefilter("always")
        store = _ingest(tmp_path, monkeypatch, dl_repr=REPR_POWERLAW,
                        prior_samples=samples, cfg=IngestConfig())
    assert not [w for w in record
                if "reject the parsed distance prior" in str(w.message)]
    assert float(_meta(store, "dL_prior_ks")) < IngestConfig().prior_ks_max


def test_ks_gate_warns_and_names_the_event_on_a_mismatch(
        tmp_path, monkeypatch):
    """Prior samples from a DIFFERENT class trip the gate, naming the event."""
    # Flat draws against a declared d^2 prior: a shape mismatch, not a bounds
    # mismatch, so it is the parse the gate is meant to reject.
    samples = np.random.default_rng(2).uniform(10.0, 10000.0, 20000)
    with pytest.warns(UserWarning, match="reject the parsed distance prior"):
        store = _ingest(tmp_path, monkeypatch, dl_repr=REPR_POWERLAW,
                        prior_samples=samples, cfg=IngestConfig())
    assert float(_meta(store, "dL_prior_ks")) > IngestConfig().prior_ks_max


def test_ks_gate_raises_when_fatal(tmp_path, monkeypatch):
    """``prior_ks_fatal=True`` turns the same mismatch into a refusal."""
    samples = np.random.default_rng(3).uniform(10.0, 10000.0, 20000)
    with pytest.raises(PriorMismatchError, match="reject the parsed distance"):
        _ingest(tmp_path, monkeypatch, dl_repr=REPR_POWERLAW,
                prior_samples=samples,
                cfg=IngestConfig(prior_ks_fatal=True))


def test_ks_is_conditioned_on_the_overlap_not_the_missing_tail(
        tmp_path, monkeypatch):
    """A CORRECT parse whose samples cover only part of the bounds passes.

    GW-37.  The model CDF was renormalized over the bounds/samples overlap
    while the empirical CDF was left unconditional, so the statistic measured
    missing tail mass rather than shape: 20000 correct PowerLaw(2) draws
    restricted to [10, 5000] against declared bounds [10, 10000] scored
    KS = 0.875 against a threshold of 0.05 -- a correct parse rejected.
    """
    samples = _power_law_draws(20000, 2.0, 10.0, 5000.0, seed=4)
    with pytest.warns(UserWarning, match="describe.*different ranges"):
        store = _ingest(tmp_path, monkeypatch, dl_repr=REPR_POWERLAW,
                        prior_samples=samples, cfg=IngestConfig())
    # The shape agrees, so the KS must not reject it ...
    assert float(_meta(store, "dL_prior_ks")) < IngestConfig().prior_ks_max
    # ... and the bounds disagreement is still reported, as ITSELF.
