"""Portable file operations used by immutable experiment exports."""
from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path


def link_or_copy(source: Path, destination: Path) -> None:
    """Link an input file, falling back to a snapshot when links are unavailable."""
    try:
        os.symlink(source, destination)
    except OSError as error:
        unsupported = error.errno in {errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP}
        unsupported = unsupported or getattr(error, "winerror", None) in {5, 1314}
        if not unsupported:
            raise
        shutil.copy2(source, destination)
