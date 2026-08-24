"""Turning a block's declaration into the number a builder writes (GW-34).

The blocks have declared their PE priors, their contexts and their safety gates
since GW-17, but the builders kept their own copy of the arithmetic: the PE
export multiplied in a hard-coded ``p_pe = m1det * p_dL_pe`` and stamped
``mass_prior_basis="uniform_detector_frame"`` on every file it wrote.  So the one
gate that mattered -- ``mass.det_pair`` refusing a prior the ``m1det`` Jacobian
does not describe -- never ran, and the 9 of 282 shipped rows whose mass prior
was never parsed were exported as verified uniform priors alongside the 273 that
were.  A declaration nothing evaluates is documentation, not a contract.

This module is the one place a builder converts a declaration into a factor.

Where each PE term is composed today
------------------------------------
* **mass** -- here, per event, in :func:`gwcat.export.pe_builder.build_pe_product`.
  Its gate is what decides whether an event may be exported at all, and its
  classification is what the file's mass-prior basis is stamped from.
* **distance** -- materialized at INGEST into the store's ``p_dL_pe`` column, by
  the same :func:`gwcat.cosmology.dL_prior_prob` the block calls.  The builder
  consumes that materialization rather than re-evaluating it: re-deriving a
  density the store already carries is how two implementations of one prior come
  to disagree, and the stored column is the one the ingest validated (GW-01/02).
* **spin** -- in the basis helpers, which additionally own the per-sample support
  accounting the flat-box and projected priors need (GW-03).
* **sky** -- in neither: ``-ln 4pi`` is declared on both sides precisely so it
  cancels, and applying it to ``p_pe`` alone would break that cancellation.
"""
from __future__ import annotations

import numpy as np


def block_prior_factor_pe(block, cols, ctx):
    """One block's PE prior as a multiplicative factor, with its gates applied.

    Prefers the block's exact linear-space ``prior_pe_factor`` over
    ``exp(ln_prior_pe)``: the two agree to an ulp, and the chieff PE export is
    contractually byte-identical to the frozen v1 exporter, so an ulp is a
    broken contract.  Both routes run the block's gates -- that is the point of
    going through the block at all.
    """
    if block.prior_pe_factor is not None:
        return block.prior_pe_factor(cols, ctx)
    if block.ln_prior_pe is None:
        raise ValueError(
            f"{block.name}: declares no PE prior, so its coordinates are not "
            f"covered by the exported density. A block in a space must carry "
            f"its own term on the PE side (see gwcat.params.block).")
    return np.exp(np.asarray(block.ln_prior_pe(cols, ctx), dtype=float))
