"""Block and space registry (GW-17).

A **space** is an ordered list of blocks. ``_CORE`` is in every space: mass,
distance and sky are always fitted, and ra/dec are unconditional -- the review
showed the non-spin coordinates were not clean either, so they get the same
declarative treatment as spin rather than being assumed correct.

This module is the single place the column set lives. Before it, adding one
coordinate meant coordinated edits across ``schema.py``, ``pe_builder.py`` (four
literals, including a dead ``_CHIEFF_COLUMNS``), ``selection_builder.py`` (three),
``writers_gwcat2.py`` (two), ``validate.py`` (three), ``catalog.py`` and
``validation_summary.py``.
"""
from __future__ import annotations

from typing import Dict, Tuple

from .block import ParameterBlock
from .blocks import (DISTANCE_DL, MASS_DET_PAIR, SKY_RADEC, SPIN_ALIGNED_Z,
                     SPIN_CARTESIAN, SPIN_CHIEFF, SPIN_CHIEFF_CHIP,
                     SPIN_CHIEFF_REFERENCE, SPIN_COMPONENT_6D,
                     SPIN_COMPONENT_POLAR, SPIN_NONE)

#: Always present, in this order.
_CORE: Tuple[ParameterBlock, ...] = (MASS_DET_PAIR, DISTANCE_DL, SKY_RADEC)

#: Every registered block, by name.
BLOCKS: Dict[str, ParameterBlock] = {
    b.name: b for b in (MASS_DET_PAIR, DISTANCE_DL, SKY_RADEC, SPIN_NONE,
                        SPIN_CHIEFF, SPIN_CHIEFF_CHIP, SPIN_CHIEFF_REFERENCE,
                        SPIN_COMPONENT_POLAR, SPIN_COMPONENT_6D,
                        SPIN_CARTESIAN, SPIN_ALIGNED_Z)
}


class ParameterSpace:
    """An ordered list of blocks plus the views the builders and validator need."""

    __slots__ = ("name", "blocks", "notes")

    def __init__(self, name: str, blocks, notes: str = ""):
        self.name = name
        self.blocks = tuple(blocks)
        self.notes = notes
        seen = {}
        for b in self.blocks:
            for c in b.columns:
                if c in seen:
                    raise ValueError(
                        f"space {name!r}: column {c!r} is claimed as a FIT "
                        f"column by both {seen[c]!r} and {b.name!r}; two "
                        f"densities would be applied to one coordinate.")
                seen[c] = b.name

    # ── column views ────────────────────────────────────────────────────────
    @property
    def fit_columns(self) -> Tuple[str, ...]:
        """Columns the exported density COVERS -- the only fittable ones."""
        out = []
        for b in self.blocks:
            out.extend(c for c in b.columns if c not in out)
        return tuple(out)

    @property
    def advisory_columns(self) -> Tuple[str, ...]:
        """Emitted but NOT covered by the density; fitting on them is an error."""
        fit = set(self.fit_columns)
        out = []
        for b in self.blocks:
            out.extend(c for c in b.advisory_columns
                       if c not in fit and c not in out)
        return tuple(out)

    @property
    def all_columns(self) -> Tuple[str, ...]:
        return self.fit_columns + self.advisory_columns

    # ── store contract ──────────────────────────────────────────────────────
    @property
    def store_required(self) -> Tuple[str, ...]:
        out = []
        for b in self.blocks:
            out.extend(p for p in b.store_required if p not in out)
        return tuple(out)

    @property
    def store_alternatives(self) -> Tuple[tuple, ...]:
        out = []
        for b in self.blocks:
            out.extend(g for g in b.store_alternatives if g not in out)
        return tuple(out)

    @property
    def store_params_fetched(self) -> Tuple[str, ...]:
        """The rng-neutrality gate: a space fetching nothing extra must consume
        an identical ``default_rng(seed)`` stream, which is what preserves the
        chieff byte-parity contract."""
        out = []
        for b in self.blocks:
            out.extend(p for p in b.store_params_fetched if p not in out)
        return tuple(out)

    # ── physics contract ────────────────────────────────────────────────────
    @property
    def is_exact(self) -> bool:
        """True when EVERY block's injection density is the campaign's actual one."""
        return all(b.exact_draw_density for b in self.blocks)

    @property
    def projections(self) -> Tuple[ParameterBlock, ...]:
        return tuple(b for b in self.blocks if b.map_kind == "projection")

    @property
    def spin_block(self) -> ParameterBlock:
        for b in self.blocks:
            if b.kind == "spin":
                return b
        return SPIN_NONE

    @property
    def mass_block(self) -> ParameterBlock:
        """The block whose PE factor the builders compose per event (GW-34)."""
        for b in self.blocks:
            if b.kind == "mass":
                return b
        return MASS_DET_PAIR

    @property
    def ranges(self) -> dict:
        out = {}
        for b in self.blocks:
            out.update(b.ranges)
        return out

    def unmet_requirements(self, campaign_meta) -> Dict[str, Tuple[str, ...]]:
        """Per-block requirements a campaign contradicts (see R1).

        Empty means nothing this space needs is contradicted.  An *unverifiable*
        campaign is not reported: "checked and failed" and "never checked" are
        different states and conflating them refuses every legacy spin-less file.
        """
        out = {}
        for b in self.blocks:
            bad = b.requires_campaign.unmet(campaign_meta)
            if bad:
                out[b.name] = bad
        return out

    def describe(self) -> dict:
        return {
            "space": self.name,
            "blocks": [b.name for b in self.blocks],
            "fit_columns": list(self.fit_columns),
            "advisory_columns": list(self.advisory_columns),
            "store_required": list(self.store_required),
            "store_params_fetched": list(self.store_params_fetched),
            "exact_draw_density": self.is_exact,
            "projections": [b.name for b in self.projections],
            "block_provenance": [b.describe() for b in self.blocks],
            "notes": self.notes,
        }

    def __repr__(self):
        return (f"ParameterSpace({self.name!r}, blocks="
                f"{[b.name for b in self.blocks]})")


#: The registered spaces.  "chieff", "chieff_chip" and "component" are the three
#: legacy spin bases and must keep matching schema.EXPORT_REQUIREMENTS.
SPACES: Dict[str, ParameterSpace] = {
    "chieff": ParameterSpace(
        "chieff", _CORE + (SPIN_CHIEFF,),
        notes="Legacy default. A PROJECTION: invalid for a campaign that is "
              "not uniform-magnitude/isotropic (GW-06)."),
    "chieff_reference": ParameterSpace(
        "chieff_reference", _CORE + (SPIN_CHIEFF_REFERENCE,),
        notes="chi_eff against a DECLARED reference spin prior, reached by "
              "REWEIGHTING the campaign's exact component draw instead of "
              "substituting for it. Buildable on any campaign -- including the "
              "ones 'chieff' must refuse -- and its a_ref must equal the PE "
              "side's ceiling. Selection side only; its PE half is a 'chieff' "
              "export at amax = a_ref."),
    "chieff_chip": ParameterSpace(
        "chieff_chip", _CORE + (SPIN_CHIEFF_CHIP,),
        notes="Opt-in, not a shipped product. Cannot be built against O4 at "
              "all, and chi_p is where the support contract bites hardest."),
    "component": ParameterSpace(
        "component", _CORE + (SPIN_COMPONENT_POLAR,),
        notes="The recommended space. Exact for ANY campaign, and measured to "
              "carry far less weight variance than chieff (ESS/nsamp median "
              "0.861 vs 0.589 on the same 259 events)."),
    "component_6d": ParameterSpace(
        "component_6d", _CORE + (SPIN_COMPONENT_6D,),
        notes="Exact; pairing two 6-D files needs an explicit acknowledgement."),
    "cartesian": ParameterSpace(
        "cartesian", _CORE + (SPIN_CARTESIAN,),
        notes="Exact but per-sample: the a^-2 factor does not cancel."),
    "aligned": ParameterSpace(
        "aligned", _CORE + (SPIN_ALIGNED_Z,),
        notes="Partial projection; needs the _single_spin_pdf 1/2 fix (GW-03)."),
    "nospin": ParameterSpace(
        "nospin", _CORE + (SPIN_NONE,),
        notes="Cosmology-only. The mitigation if a 4-D spin population ever "
              "does collapse N_eff."),
}

#: The three spaces that correspond to an existing ``spin_basis`` value.
LEGACY_SPIN_BASES = ("chieff", "chieff_chip", "component")


def get_space(name: str) -> ParameterSpace:
    if name not in SPACES:
        raise KeyError(
            f"unknown parameter space {name!r}; registered spaces are "
            f"{sorted(SPACES)}")
    return SPACES[name]


def list_spaces() -> Tuple[str, ...]:
    return tuple(sorted(SPACES))


#: The default parameter space for EVERY no-argument export path -- the CLI's
#: ``export pe`` / ``export selection``, ``GWCatalog.export``, and
#: ``SelectionSet.export`` / ``CombinedSelectionSet.export``.
#:
#: It lives here, in the registry, because the PE and selection defaults used to
#: be written out separately and had drifted apart: PE defaulted to ``chieff``
#: and selection to ``component``, so *either* no-argument path -- CLI or Python
#: -- produced a pair that fails the cross-file basis check by construction. A
#: default that cannot be used with itself is not a default. One constant, in
#: the module that owns the spaces, is what stops them drifting again.
#:
#: ``component`` because it is the exact (bijective) space: its density is
#: definable against any campaign, whereas a projection is valid only against a
#: uniform-magnitude/isotropic parent draw (R1) and silently wrong otherwise.
#: The legacy chi_eff behaviour remains one explicit argument away, and
#: ``GWCatalog.to_darksirens`` (frozen v1) is unaffected.
DEFAULT_PARAMETER_SPACE = "component"
