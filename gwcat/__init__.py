"""gwcat: fast preprocessing of GWTC-2.1/3/4/5 cosmo PE files.

Three stages:
  0. gwcat.fetch.fetch_catalog(...)   download PE files from Zenodo
  1. gwcat.ingest.build_store(...)    raw PESummary cosmo files -> store.h5
  2. gwcat.catalog.GWCatalog(...)     fast cuts + derived quantities + export

Or all at once:
  gwcat.fetch.fetch_and_build(["GWTC-2.1", "GWTC-3", "GWTC-4.1", "GWTC-5"], out="store.h5")

The (m1det, q, dL)-basis mass Jacobian is applied ONLY in
GWCatalog.to_darksirens; the store stays mass-prior-agnostic.
(_to_darksirens_format is a deprecated alias kept for compatibility.)
"""
__version__ = "0.1.0"

from .catalog import GWCatalog, validate_export
from .ingest import (build_store, merge_store, merge_stores, inspect,
                     IngestConfig, DEFAULT_PARAMS)
from .selection import SelectionSet, CombinedSelectionSet
from .source_class import (SOURCE_CLASSES, SourceClassMeta,
                          normalize_source_class, resolve_filter_classes,
                          load_event_list)
from .schema import (PARAMETER_GROUPS, EXPORT_REQUIREMENTS, DARKSIRENS_REQUIRED,
                    MissingParameterError, required_params)
from .waveform_policy import WAVEFORM_POLICIES, resolve_policy

# fetch has optional deps (requests, tqdm); import lazily
def fetch_and_build(*args, **kwargs):
    """Convenience re-export. See gwcat.fetch.fetch_and_build."""
    from .fetch import fetch_and_build as _fab
    return _fab(*args, **kwargs)

def fetch_catalog(*args, **kwargs):
    """Convenience re-export. See gwcat.fetch.fetch_catalog."""
    from .fetch import fetch_catalog as _fc
    return _fc(*args, **kwargs)

__all__ = [
    "GWCatalog", "SelectionSet", "CombinedSelectionSet",
    "build_store", "merge_store", "merge_stores", "inspect", "IngestConfig",
    "DEFAULT_PARAMS",
    "validate_export",
    "SOURCE_CLASSES", "SourceClassMeta", "normalize_source_class",
    "resolve_filter_classes", "load_event_list",
    "PARAMETER_GROUPS", "EXPORT_REQUIREMENTS", "DARKSIRENS_REQUIRED",
    "MissingParameterError", "required_params",
    "WAVEFORM_POLICIES", "resolve_policy",
    "fetch_and_build", "fetch_catalog",
    "__version__",
]