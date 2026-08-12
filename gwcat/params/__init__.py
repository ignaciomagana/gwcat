"""Declarative fit-parameter spaces for gwcat (GW-17, design PR-A).

Pure assembly: every callable here is read out of the existing modules, there is
no new physics, and **nothing in gwcat imports this package yet**.  Wiring the
builders to it is GW-18 (PE) and GW-19 (selection).

    from gwcat.params import SPACES, get_space
    sp = get_space("component")
    sp.fit_columns          # ('m1det','m2det','dL','ra','dec','a1','a2',...)
    sp.advisory_columns     # ('redshift','chieff','chip')  -- do NOT fit these
    sp.is_exact             # True: no assumed density anywhere in the space
"""
from .block import (KINDS, MAP_KINDS, CampaignRequirement, ParameterBlock,
                    Range)
from .context import InjContext, PEContext
from .registry import (BLOCKS, LEGACY_SPIN_BASES, SPACES, ParameterSpace,
                       get_space, list_spaces)

__all__ = ["ParameterBlock", "Range", "CampaignRequirement", "KINDS",
           "MAP_KINDS", "PEContext", "InjContext", "BLOCKS", "SPACES",
           "ParameterSpace", "get_space", "list_spaces", "LEGACY_SPIN_BASES"]
