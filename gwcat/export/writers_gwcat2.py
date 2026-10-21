"""gwcat2 PE writer for the versioned export pipeline (PR 3).

Owns ONLY serialization: it turns an :class:`~gwcat.export.product.ExportProduct`
(built by :func:`gwcat.export.pe_builder.build_pe_product`) into a
``format_version="gwcat-pe-2.0"`` HDF5 file.  All physics is upstream in the
builder; this module never touches columns or provenance semantics beyond
stamping the format version and the chieff-basis legacy-compat spin attrs.

Two properties every writer here holds (GW-30):

  * a file is written ONCE, complete, at the version it will keep.  The 2.1
    writers used to delegate to the 2.0 writer -- publishing a file that
    identified as 2.0, generating its validation summary against that file, and
    only then reopening it to stamp 2.1 and the contract attrs.  The summary was
    therefore computed from a file that did not yet exist in its final form, and
    a failure between the two steps left a complete-looking 2.0 file at the
    destination.  Both versions now go through one code path that stamps the
    version and its version-specific attrs in the same open file, before
    anything reads it back;
  * the write is ATOMIC (:func:`gwcat.export.atomic.atomic_output_path`): a
    sibling temp file, fsynced, then renamed onto the destination.  An
    interrupted export cannot leave a truncated HDF5 file where a consumer
    expects a complete one, and cannot destroy the previous good export.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from typing import Optional

import h5py
import numpy as np

from .atomic import atomic_output_path
from .registry import register_exporter

#: The datasets a chieff-basis PE file writes, in legacy order.
_PE_DATASETS = ["ra", "dec", "m1det", "m2det", "chieff", "dL", "p_pe",
                "redshift", "m1src", "m2src"]

#: The legacy selection datasets, in the order the v1 exporter writes them,
#: followed by the additive component-spin columns.
_SELECTION_DATASETS = ["m1det", "m2det", "q", "dL", "chieff", "ra", "dec",
                       "m1src", "m2src", "redshift", "pdraw",
                       "a1", "a2", "cost1", "cost2", "chip"]

#: The format versions this module writes.  2.0 is the default everywhere; 2.1
#: is opt-in (see the note above the 2.1 writers for why, and what a 2.0 pair
#: does not get checked for).
_PE_FORMAT_20 = "gwcat-pe-2.0"
_SEL_FORMAT_20 = "gwcat-selection-2.0"
_PE_FORMAT_21 = "gwcat-pe-2.1"
_SEL_FORMAT_21 = "gwcat-selection-2.1"


# --------------------------------------------------------------------------
# Shared serialization pieces (one implementation per concern, used by both
# format versions -- a 2.1 file is a 2.0 file plus the contract attrs, and it
# must not be produced by mutating a published 2.0 file).
# --------------------------------------------------------------------------
def _stamp_attrs(f, product, format_version):
    """Provenance attrs verbatim, then the version and the writer's identity."""
    for k, v in product.attrs.items():
        f.attrs[k] = v

    # Format version is the writer's, never the builder's.
    f.attrs["format_version"] = format_version

    # Which gwcat wrote this file (GW-22 rider / DS-10): the commit is the
    # provenance that matters for an editable install, where the version
    # string does not move between commits.  "unknown" for a non-git
    # install; "-dirty" when the worktree had uncommitted gwcat changes.
    from ..validation_summary import gwcat_commit, package_version
    f.attrs["writer_commit"] = gwcat_commit()
    f.attrs["writer_version"] = package_version()
    f.attrs["writer_environment"] = _writer_environment()


def _writer_environment():
    """The run environment that sets an export's last bits, as a JSON string.

    Review 2026-10-20, Low 4: a Haswell rebuild of the production PE file
    differs from the AVX-512 build in about 378k p_pe values at the 1e-17
    level, so a bitwise rebuild check cannot tell environment from code unless
    the file says which environment wrote it: the library versions, numpy's
    BLAS, its compiled-in SIMD baseline and the dispatch targets this CPU runs.
    Never raises; a field that cannot be read is "unknown".
    """
    import json as _json
    import platform
    import sys

    env = {"python": sys.version.split()[0], "machine": platform.machine(),
           "numpy": np.__version__, "h5py": h5py.__version__,
           "hdf5": h5py.version.hdf5_version}
    for mod in ("scipy", "astropy"):
        try:
            env[mod] = str(__import__(mod).__version__)
        except Exception:
            env[mod] = "unknown"
    try:
        try:
            from numpy._core import _multiarray_umath as _mu
        except ImportError:                      # numpy < 2
            from numpy.core import _multiarray_umath as _mu
        feats = _mu.__cpu_features__
        env["numpy_cpu_baseline"] = list(_mu.__cpu_baseline__)
        env["numpy_cpu_dispatch_active"] = [
            t for t in _mu.__cpu_dispatch__ if feats.get(t)]
    except Exception:
        env["numpy_cpu_baseline"] = "unknown"
        env["numpy_cpu_dispatch_active"] = "unknown"
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        env["numpy_blas"] = " ".join(
            str(blas.get(k, "")) for k in ("name", "version",
                                           "openblas configuration")).strip()
    except Exception:
        env["numpy_blas"] = "unknown"
    return _json.dumps(env, sort_keys=True)


def _stamp_legacy_spin_attrs(f, product):
    """The gwcat-1.0 spin contract, stated on every basis.

    chieff keeps the historical values; every other basis states the truth
    explicitly rather than omitting them (GW-21).  Two reasons the omission was
    not safe:

      * darksirens REQUIRES chi_eff_in_p_pe on a gwcat-pe-2.0 file
        (gw/utils.py required-attrs list), so a file lacking it does not fail a
        physics check -- it fails to load, with a member-check error that says
        nothing about the basis;
      * "absent" and "False" are different claims. A consumer must be able to
        distinguish "no chi_eff prior was applied" from "this file does not
        say", and only the first is a statement it can act on.
    """
    for k, v in _legacy_spin_flags(product.spin_basis).items():
        f.attrs[k] = v


def _legacy_spin_flags(spin_basis):
    """The gwcat-1.0 spin attrs :func:`_stamp_legacy_spin_attrs` writes.

    One definition, so the PE summary can state the same values as the file
    (review 2026-10-20, Low 42: a chieff_chip summary said
    chi_eff_prior_applied_to_p_pe=True beside a file that says False).
    """
    if spin_basis == "chieff":
        return {"spin_prior_mode": "include",
                "chi_eff_prior_applied_to_p_pe": True,
                "chi_eff_in_p_pe": True}
    mode = {"component": "component_flat",
            "chieff_chip": "chieff_chip_joint",
            "nospin": "none"}.get(spin_basis, "unknown")
    # The 1-D chi_eff prior specifically is NOT in p_pe for any of
    # these: component carries a flat box, chieff_chip a JOINT
    # (chi_eff, chi_p) density, nospin nothing at all.  So the legacy
    # scalar flag is False in every case, and the basis-specific attrs
    # written by the builder are what say what WAS applied.
    return {"spin_prior_mode": mode,
            "chi_eff_prior_applied_to_p_pe": False,
            "chi_eff_in_p_pe": False}


def _write_columns(f, columns, order, **create_kwargs):
    """Write ``order`` first (stable), then any extra columns a basis added."""
    written = set()
    for k in order:
        if k in columns:
            f.create_dataset(k, data=columns[k], **create_kwargs)
            written.add(k)
    for k, arr in columns.items():
        if k not in written:
            f.create_dataset(k, data=arr, **create_kwargs)


#: The sidecars :func:`gwcat.validation_summary.write_validation_summary` writes
#: next to an export.
_SUMMARY_SUFFIXES = (".validation_summary.json", ".validation_summary.md")


def _remove_stale_summary(out_path):
    """Remove a previous export's summary before the new file is published.

    Review 2026-10-20, Medium 53: a summary is written only when asked for (the
    Python default is not to), and nothing removed an old one, so a re-export
    to the same path -- from Python, or with ``--no-summary`` -- sat beside a
    summary listing the previous product's events and nsamp under the same
    output_path.  Called before the publishing rename, so at no moment does the
    new file sit beside the old summary; no summary is better than a wrong one.
    """
    for suffix in _SUMMARY_SUFFIXES:
        p = os.fspath(out_path) + suffix
        if os.path.lexists(p):
            os.remove(p)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_summary(product, out_path, format_version, summary_context):
    """Write the validation summary next to the FINISHED file at ``out_path``.

    Called only after the atomic rename, so everything that reads the file back
    (:func:`gwcat.validation_summary.v2_export_summary_additions`) sees the file
    the consumer will see -- final version, final attrs.  The summary carries
    the sha256 of that file, so a summary can be checked against the file
    beside it rather than trusted by its name.
    """
    from ..validation_summary import write_validation_summary
    summary = dict(product.summary)
    summary["output_path"] = str(out_path)
    summary["output_sha256"] = _sha256(out_path)
    # The version the summary actually describes.  A 2.1 export's summary used
    # to be indistinguishable from a 2.0 one because it was generated while the
    # file still said 2.0.
    summary["format_version"] = format_version
    if product.kind == "pe":
        # The legacy spin flags as the FILE states them (Low 42), not the
        # builder's chi_eff_included, which is True for chieff_chip's joint
        # (chi_eff, chi_p) density.
        flags = _legacy_spin_flags(product.spin_basis)
        for k in ("spin_prior_mode", "chi_eff_prior_applied_to_p_pe"):
            summary[k] = flags[k]
    if summary_context:
        summary.update(summary_context)
    write_validation_summary(out_path, summary)


def _refuse_undeclared_datasets(product):
    """Every dataset is a declared fit/advisory column or a weight (review
    2026-10-20); an undeclared one is a coordinate a consumer cannot place."""
    from ..params import get_space
    extra = get_space(product.spin_basis).undeclared_datasets(product.columns)
    if extra:
        raise ValueError(
            f"{product.kind} product (spin_basis={product.spin_basis!r}) would "
            f"write dataset(s) {list(extra)} that parameter space "
            f"{product.spin_basis!r} declares neither as fit nor as advisory "
            f"columns. Declare them in the block registry (gwcat/params).")


def _write_pe(product, out_path, format_version, *, with_contract=False,
              write_summary=False, summary_context=None):
    """Serialize a PE product at ``format_version`` (2.0 or 2.1)."""
    if product.kind != "pe":
        raise ValueError(
            f"write_pe_gwcat2 expects a kind='pe' product, got "
            f"kind={product.kind!r}.")
    _refuse_undeclared_datasets(product)

    extra_attrs = _contract_attrs(product, "pe") if with_contract else {}

    with atomic_output_path(out_path) as tmp_path:
        with h5py.File(tmp_path, "w") as f:
            _stamp_attrs(f, product, format_version)
            _stamp_legacy_spin_attrs(f, product)
            for k, v in extra_attrs.items():
                f.attrs[k] = v
            # Datasets (gzip, like the legacy exporter).
            _write_columns(f, product.columns, _PE_DATASETS,
                           compression="gzip", shuffle=False)
        _remove_stale_summary(out_path)

    if write_summary:
        _write_summary(product, out_path, format_version, summary_context)

    return str(out_path)


def _warn_if_core_cannot_load(product):
    """Say at write time that darksirens-core will refuse this selection file.

    Review 2026-10-20, Medium 13.  The cumulative mixtures ship no sky position
    for their semianalytic O1/O2 rows, so every route out of them reaches core
    without one: ``sky_marginal=True`` drops ra/dec (core's selection contract
    requires both datasets), and the default writes NaN ra/dec that core
    refuses through ``sky_position_available``.  No wrong number reaches core --
    the refusal is loud -- but it comes at load time, after an export and a
    ``gwcat validate --strict`` that both passed.  A sky-marginal file can serve
    only a sky-independent (spectral-siren) fit in a consumer that accepts one;
    a galaxy-catalogue selection integral needs every injection's sky position.
    Making such a file loadable needs a sky-marginal contract agreed with core.
    """
    sky = np.asarray(product.attrs.get("sky_position_available", True),
                     dtype=bool).ravel()
    marginal = bool(product.attrs.get("sky_marginalized", False))
    if not marginal and sky.all():
        return
    why = ("sky_marginalized=True: no ra/dec columns" if marginal else
           f"sky_position_available per campaign = {sky.tolist()}: NaN ra/dec "
           f"rows")
    warnings.warn(
        f"this selection product has no sky position for every injection "
        f"({why}). darksirens-core requires finite ra/dec on every injection "
        f"and will refuse to load it; no gwcat format makes it loadable there "
        f"today. It can serve only a sky-independent fit in a consumer that "
        f"accepts a sky-marginal selection.")


def _write_selection(product, out_path, format_version, *, with_contract=False,
                     write_summary=False, summary_context=None):
    """Serialize a selection product at ``format_version`` (2.0 or 2.1)."""
    if product.kind != "selection":
        raise ValueError(
            f"write_selection_gwcat2 expects a kind='selection' product, got "
            f"kind={product.kind!r}.")
    _refuse_undeclared_datasets(product)

    if with_contract and bool(product.attrs.get("sky_marginalized", False)):
        # GW-39.  The 2.1 contract declares the space's fit_columns, which put
        # ra/dec in the fit (sky.radec, -ln 4pi on both sides); a
        # sky-marginalised product has no ra/dec, so its contract would
        # describe coordinates the file does not contain.
        raise ValueError(
            "a sky-marginalised selection product (sky_marginalized=True, no "
            "ra/dec columns) cannot be written as gwcat-selection-2.1: the 2.1 "
            "contract's fit_columns include ra/dec. Write it with the default "
            "'gwcat2' (2.0) format.")
    if with_contract:
        # Review 2026-10-20: the contract declares the space's fit columns, so
        # a column with non-finite rows (ra/dec of any campaign that drew no
        # sky) must not be declared fittable -- it was written and validated.
        from ..params import get_space
        get_space(product.spin_basis).refuse_nonfinite_fit_columns(
            product.columns, context=f"{format_version} writer")
    _warn_if_core_cannot_load(product)
    extra_attrs = _contract_attrs(product, "selection") if with_contract else {}

    with atomic_output_path(out_path) as tmp_path:
        with h5py.File(tmp_path, "w") as f:
            _stamp_attrs(f, product, format_version)
            for k, v in extra_attrs.items():
                f.attrs[k] = v
            # Datasets (gzip, like the legacy selection exporter).
            _write_columns(f, product.columns, _SELECTION_DATASETS,
                           compression="gzip")
        _remove_stale_summary(out_path)

    if write_summary:
        _write_summary(product, out_path, format_version, summary_context)

    return str(out_path)


# --------------------------------------------------------------------------
# Registered writers
# --------------------------------------------------------------------------
@register_exporter("gwcat2", kind="pe")
def write_pe_gwcat2(product, out_path, *, write_summary: bool = False,
                    summary_context: Optional[dict] = None):
    """Serialize a PE :class:`ExportProduct` as a ``gwcat-pe-2.0`` HDF5 file.

    Columns are written as gzip-compressed datasets (as the legacy exporter
    does); ``product.attrs`` is written verbatim; ``format_version`` is set to
    ``"gwcat-pe-2.0"``; and the legacy-compat spin attrs (``spin_prior_mode``,
    ``chi_eff_prior_applied_to_p_pe`` and its ``chi_eff_in_p_pe`` alias) are
    stated for every basis so downstream tooling that reads the gwcat-1.0 spin
    contract keeps working.  The file is written atomically (temp sibling +
    rename), so a failed export never replaces a good one.

    Parameters
    ----------
    product : gwcat.export.product.ExportProduct
        A ``kind="pe"`` product from :func:`build_pe_product`.
    out_path : str or path-like
        Destination HDF5 path.
    write_summary : bool, default False
        When True, also write ``<out_path>.validation_summary.json`` (+ ``.md``)
        next to ``out_path`` from ``product.summary`` (see
        :mod:`gwcat.validation_summary`).
    summary_context : dict, optional
        Extra fields merged into the written summary (last, so they win).

    Returns
    -------
    str
        ``str(out_path)``.
    """
    return _write_pe(product, out_path, _PE_FORMAT_20,
                     write_summary=write_summary,
                     summary_context=summary_context)


@register_exporter("gwcat2", kind="selection")
def write_selection_gwcat2(product, out_path, *, write_summary: bool = False,
                           summary_context: Optional[dict] = None):
    """Serialize a selection :class:`ExportProduct` as ``gwcat-selection-2.0``.

    Owns ONLY serialization: gzip-compressed datasets (legacy order first, then
    any extra spin columns), ``product.attrs`` verbatim, and
    ``format_version="gwcat-selection-2.0"``, written atomically.  All physics
    -- the detection cut, source-class subsetting, the per-basis spin factor and
    the Essick fractions -- is upstream in
    :func:`gwcat.export.selection_builder.build_selection_product`.

    Parameters
    ----------
    product : gwcat.export.product.ExportProduct
        A ``kind="selection"`` product from ``build_selection_product``.
    out_path : str or path-like
        Destination HDF5 path.
    write_summary : bool, default False
        When True, also write ``<out_path>.validation_summary.json`` (+ ``.md``)
        next to ``out_path`` from ``product.summary`` (the legacy selection
        exporter's behavior, via :mod:`gwcat.validation_summary`).
    summary_context : dict, optional
        Extra fields merged into the written summary (last, so they win).

    Returns
    -------
    str
        ``str(out_path)``.
    """
    return _write_selection(product, out_path, _SEL_FORMAT_20,
                            write_summary=write_summary,
                            summary_context=summary_context)


# ==========================================================================
# Format 2.1 (GW-22) -- OPT-IN.  "gwcat2" still means 2.0.
# ==========================================================================
# 2.1 adds the declarative contract to the file: the parameter space, its fit
# and advisory columns, per-block provenance, and a contract_hash that makes a
# mismatched PE/selection pairing one comparison instead of a growing list of
# per-field checks.
#
# It is registered as a SEPARATE format name rather than bumping "gwcat2",
# because `_require_hdf5_format` in darksirens is a closed literal tuple: a 2.1
# file is unloadable by any consumer that has not been taught the version, and
# the plan's rule is that the consumer patch deploys FIRST.
#
# That reason is out of date (review 2026-10-20, Medium 2): darksirens-core has
# read 2.1 since 2026-09-09.  The reason 2.0 is STILL the default is the cost of
# keeping it: a 2.0 file carries no contract_hash, and core's
# require_matching_contract skips any pair in which either file lacks one, so
# the production pair (chieff PE with chieff_reference selection) is loaded
# with no pairing check at all.  Switching the default does not fix that yet:
# the 2.1 hash covers parameter_space and advisory_columns, which always differ
# for that designed pair (Medium 10), so a 2.1 rebuild of it is refused by
# `gwcat validate` and by core.  The order is: make the hash accept the
# reference pair (and carry the PE-ceiling = a_ref equality it needs), then
# make 2.1 the default and rebuild the pair, then have core refuse a pair with
# no hash unless explicitly allowed.  Until then 2.0 stays the default so the
# production pair keeps loading.


def _contract_attrs(product, kind):
    """The 2.1 contract attrs for a product, plus its hash."""
    import json as _json

    from ..params import get_space
    from .contract import (build_contract, contract_hash,
                           selection_contract_fields)

    space = get_space(product.spin_basis)
    a = product.attrs
    spin = space.spin_block

    # The detection cut, as a (statistic, threshold) pair rather than a bare
    # number: Essick & Fishbach require the EVENT cut and the INJECTION cut to
    # be the same statistic at the same threshold, and comparing thresholds
    # without their statistic is how that check stayed vacuous.
    # Name the STATISTIC, not the side's attr: "far_max" and "far_threshold" are
    # two spellings of one cut, and recording the spellings made the two sides
    # differ by construction.
    thr = a.get("far_max") if kind == "pe" else a.get("far_threshold")
    thr = None if thr is None or not np.isfinite(float(thr)) else float(thr)
    # From the product's OWN attrs, not from the presence of a FAR threshold
    # (GW-37).  When --snr-threshold is set the selection builder widens the
    # injection mask to `far-detected OR snr > t` and records
    # significance_type="far_or_snr", which these two lines ignored: the file
    # then declared itself FAR-only and hash-matched a genuinely FAR-only PE
    # file, while its injection mask was strictly looser than the event cut --
    # mu biased high, the inferred rate biased low, and nothing to see in either
    # file.  Two lines below this module's own comment saying exactly that.
    snr_min = a.get("snr_min") if kind == "pe" else a.get(
        "significance_snr_threshold")
    snr_min = (None if snr_min is None or not np.isfinite(float(snr_min))
               else float(snr_min))
    if kind == "pe":
        # The PE side's cuts are select()'s, so name THEM (review 2026-10-20,
        # Medium 28): FAR on the catalogue FAR, SNR on the POSTERIOR-MEDIAN SNR,
        # and the two combined as an AND.  Inferring the name from which attrs
        # exist made far_max + snr_min read 'far_or_snr' -- the name of the
        # selection builder's OR-widened INJECTION mask, a strictly looser cut
        # no PE file can apply -- so the two contracts agreed on statistic,
        # threshold and snr_min while describing different selections.
        stat = ("far_and_snr_med" if thr is not None and snr_min is not None
                else "far" if thr is not None
                else "snr_med" if snr_min is not None else None)
    else:
        stat = a.get("significance_type")
        stat = (str(stat) if stat is not None
                else (None if thr is None
                      else ("far_or_snr" if snr_min is not None else "far")))
        if stat is None and snr_min is not None:
            stat = "snr"

    _sel_fields = dict(selection_contract_fields(a))
    # z_max is an EXPORT argument, not a select() cut, so it lives in the attrs
    # rather than in the selection_spec JSON the helper reads (GW-37).
    _zmax = a.get("z_max")
    _sel_fields["z_max"] = (None if _zmax is None
                            or not np.isfinite(float(_zmax)) else float(_zmax))
    if kind == "selection" and snr_min is not None:
        # `selection_contract_fields` reads snr_min out of the PE-side
        # selection_spec, which an injection product does not have, so a mask
        # widened by --snr-threshold left the field None and the SNR leg could
        # not be compared at all.  State the threshold this file actually used.
        _sel_fields["snr_min"] = snr_min

    contract = build_contract(
        parameter_space=space.name,
        fit_columns=list(space.fit_columns),
        advisory_columns=list(space.advisory_columns),
        spin_basis_kind=spin.map_kind,
        spin_density_exact=bool(space.is_exact),
        mass_prior_kind=a.get("mass_prior_basis") or a.get("mass_prior_kind"),
        # The PE builder's label over the store's per-row class (review
        # 2026-10-20, Medium 43); None on the selection side and for a store
        # that records no class.
        dL_prior_kind=a.get("dL_prior_kind"),
        # What the densities CONTAIN (review 2026-10-20, Low 67).  sky.radec
        # declares -ln 4pi on both sides precisely so it cancels, and neither
        # builder multiplies it into p_pe or pdraw (gwcat.params.compose: the
        # sky is "in neither"), so the hard-coded True stated a factor neither
        # file carries.  A builder that ever applies it must stamp the attr.
        sky_prior_in_density=bool(a.get("sky_prior_in_density", False)),
        source_class=a.get("source_class_filter") or None,
        detection_statistic=stat,
        detection_threshold=thr,
        allow_missing_far=bool(a.get("allow_missing_far", False)),
        cosmology_H0=(a.get("pe_cosmology_H0") if kind == "pe"
                      else a.get("cosmology_H0")),
        cosmology_Om0=(a.get("pe_cosmology_Om0") if kind == "pe"
                       else a.get("cosmology_Om0")),
        # Every OTHER effective event cut, plus the two selection digests
        # (GW-33).  The class filter used to be the only cut in the contract, so
        # a file cut on p_astro / a name whitelist / median masses could state a
        # matching source_class and pass.
        **_sel_fields,
    )
    return {
        "parameter_space": space.name,
        "parameter_blocks": np.array([b.name for b in space.blocks],
                                     dtype=h5py.string_dtype()),
        "fit_columns": np.array(list(space.fit_columns),
                                dtype=h5py.string_dtype()),
        "advisory_columns": np.array(list(space.advisory_columns),
                                     dtype=h5py.string_dtype()),
        "spin_basis_kind": spin.map_kind,
        "spin_density_exact": bool(space.is_exact),
        "block_provenance": _json.dumps(space.describe()),
        "contract": _json.dumps(contract),
        "contract_hash": contract_hash(contract),
    }


@register_exporter("gwcat2.1", kind="pe")
def write_pe_gwcat21(product, out_path, *, write_summary: bool = False,
                     summary_context=None):
    """Write a ``gwcat-pe-2.1`` PE export (opt-in; see the module note).

    The file carries ``format_version="gwcat-pe-2.1"`` and the contract attrs
    from the first byte -- it is never published as 2.0 and upgraded.
    """
    return _write_pe(product, out_path, _PE_FORMAT_21, with_contract=True,
                     write_summary=write_summary,
                     summary_context=summary_context)


@register_exporter("gwcat2.1", kind="selection")
def write_selection_gwcat21(product, out_path, *, write_summary: bool = False,
                            summary_context=None):
    """Write a ``gwcat-selection-2.1`` selection export (opt-in).

    As with the PE writer, 2.1 and its contract attrs are stamped in the same
    open file as everything else, not patched onto a published 2.0 file.
    """
    return _write_selection(product, out_path, _SEL_FORMAT_21,
                            with_contract=True, write_summary=write_summary,
                            summary_context=summary_context)
