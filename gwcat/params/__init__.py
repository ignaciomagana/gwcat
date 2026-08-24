"""Declarative fit-parameter spaces for gwcat (GW-17, design PR-A).

The blocks declare what an export fits and under what density; the builders
compose those declarations rather than carrying their own copy of the arithmetic
(GW-18 wired the requirements, GW-34 the PE mass density and its gate -- see
:mod:`gwcat.params.compose` for which term is composed where).

    from gwcat.params import SPACES, get_space
    sp = get_space("component")
    sp.fit_columns          # ('m1det','q','dL','ra','dec','a1','a2',...)
    sp.advisory_columns     # ('m2det','redshift','chieff','chip') -- do NOT fit
    sp.is_exact             # True: no assumed density anywhere in the space
"""
from .block import (KINDS, MAP_KINDS, CampaignRequirement, ParameterBlock,
                    Range)
from .compose import block_prior_factor_pe
from .context import InjContext, PEContext
from .registry import (BLOCKS, DEFAULT_PARAMETER_SPACE, LEGACY_SPIN_BASES,
                       SPACES, ParameterSpace, get_space, list_spaces)

__all__ = ["ParameterBlock", "Range", "CampaignRequirement", "KINDS",
           "MAP_KINDS", "PEContext", "InjContext", "BLOCKS", "SPACES",
           "ParameterSpace", "get_space", "list_spaces", "LEGACY_SPIN_BASES",
           "DEFAULT_PARAMETER_SPACE", "block_prior_factor_pe"]
