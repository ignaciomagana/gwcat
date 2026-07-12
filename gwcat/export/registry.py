"""Writer registry for the versioned export pipeline (PR 3).

Writers own only serialization; each is registered against a ``(format, kind)``
pair -- e.g. ``("gwcat2", "pe")`` -- so a single format name can carry distinct
writers for posterior-sample (``"pe"``) and selection-function (``"selection"``)
products.  Builders (which own the physics) look up a writer via
:func:`get_exporter`.

The registry is deliberately tiny and side-effect-driven: importing
:mod:`gwcat.export` imports the writer modules, whose module-level
``@register_exporter(...)`` decorators populate this table.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

#: ``(format_name, kind) -> writer_callable``.
_REGISTRY: Dict[Tuple[str, str], Callable] = {}


def register_exporter(name: str, kind: str) -> Callable[[Callable], Callable]:
    """Decorator registering a writer for ``(name, kind)``.

    Parameters
    ----------
    name : str
        Format name (e.g. ``"gwcat2"``).
    kind : str
        Product kind the writer serializes (``"pe"`` or ``"selection"``).

    Raises
    ------
    ValueError
        If a writer is already registered for the same ``(name, kind)`` -- a
        duplicate registration is a programming error, never silently
        overridden.
    """

    def _decorator(func: Callable) -> Callable:
        key = (name, kind)
        if key in _REGISTRY:
            raise ValueError(
                f"an exporter is already registered for format={name!r}, "
                f"kind={kind!r}; duplicate registration is not allowed.")
        _REGISTRY[key] = func
        return func

    return _decorator


def get_exporter(name: str, kind: str) -> Callable:
    """Return the writer registered for ``(name, kind)``.

    Raises
    ------
    ValueError
        If no writer is registered for ``(name, kind)``; the message lists the
        known formats for that kind (and, if none, all known formats).
    """
    key = (name, kind)
    if key not in _REGISTRY:
        known_for_kind = sorted(n for (n, k) in _REGISTRY if k == kind)
        if known_for_kind:
            known = f"known {kind!r} formats: {known_for_kind}"
        else:
            known = f"known formats: {list_formats()}"
        raise ValueError(
            f"unknown export format={name!r} for kind={kind!r}; {known}.")
    return _REGISTRY[key]


def list_formats() -> List[Tuple[str, str]]:
    """Return a sorted list of registered ``(format_name, kind)`` pairs."""
    return sorted(_REGISTRY.keys())
