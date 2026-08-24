"""Atomic writes for scientific output files (GW-30).

Every file this package writes is an input to somebody's inference run, and a
half-written one is worse than a missing one: a truncated HDF5 export, or an
export whose ``format_version`` was stamped before the attrs that version
promises, is read by a consumer as if it were complete.  Both were reachable --
the writers opened the FINAL path with ``h5py.File(path, "w")`` (which truncates
an existing good file on the first byte), and the 2.1 writers went further,
publishing a complete-looking ``gwcat-pe-2.0`` file at the destination and only
then reopening it to stamp 2.1 and its contract.

So a scientific output lands the way :mod:`gwcat.fetch_cache` already lands a
cache record: serialized to a sibling temp file in the destination directory,
flushed to disk, then ``os.replace``d onto the final path -- one rename, either
the old file or the new one, never a mixture.  The temp file is a sibling (not
in ``/tmp``) so the rename is within one filesystem and therefore atomic.
"""
from __future__ import annotations

import contextlib
import os

__all__ = ["atomic_output_path"]


def _fsync_file(path: str) -> None:
    """Flush ``path``'s contents to the storage device."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(directory: str) -> None:
    """Flush the directory entry so the rename itself survives a crash.

    Best-effort: some filesystems refuse to open or fsync a directory, and the
    ordering that matters (contents on disk BEFORE the rename that publishes
    them) is already guaranteed by :func:`_fsync_file`.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@contextlib.contextmanager
def atomic_output_path(out_path):
    """Context manager yielding a temp path to write ``out_path``'s content to.

    On clean exit the temp file is fsynced and ``os.replace``d onto
    ``out_path``; on any exception (including ``KeyboardInterrupt``) it is
    removed and ``out_path`` is left exactly as it was.

    Parameters
    ----------
    out_path : str or path-like
        The final destination.  Its directory must exist (the temp file is
        created there, so the publishing rename never crosses a filesystem).

    Yields
    ------
    str
        The path to write.  Callers must have finished writing and CLOSED the
        file before the block exits -- nothing is published until then.

    Examples
    --------
    >>> with atomic_output_path(out) as tmp:      # doctest: +SKIP
    ...     with h5py.File(tmp, "w") as f:
    ...         f.attrs["format_version"] = "gwcat-pe-2.1"
    """
    final = os.fspath(out_path)
    directory = os.path.dirname(os.path.abspath(final))
    # Hidden, pid-tagged sibling: two processes writing the same output do not
    # collide on the temp file, and a leftover (SIGKILL, node death) is visibly
    # not the product.
    tmp = os.path.join(
        directory, f".{os.path.basename(final)}.{os.getpid()}.gwcat-tmp")
    try:
        yield tmp
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:                      # nothing better to do here
                pass
        raise
    _fsync_file(tmp)
    os.replace(tmp, final)
    _fsync_dir(directory)
