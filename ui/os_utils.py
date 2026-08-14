from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def open_path(path: str) -> None:
    """Open a file or folder with the OS's default handler.

    Uses Popen (fire-and-forget) rather than run/call: xdg-open in
    particular can block for as long as the launched application stays
    open, which would freeze the GUI thread since this is called directly
    from a button handler, not a worker thread.
    """
    p = Path(path)
    if sys.platform == "darwin":
        cmd = "open"
    elif sys.platform.startswith("win"):
        cmd = "explorer"
    else:
        cmd = "xdg-open"
    # Resolve the full executable path ourselves instead of handing Popen a
    # bare command name to look up on PATH at call time.
    exe = shutil.which(cmd) or cmd
    # `path` is always this plugin's own already-written report file or
    # output directory (see callers: sdm_dock.py/summary_page.py, both pass
    # a RunResult path from a run this session just performed) — never
    # externally supplied text — and is passed as an argv list, never
    # through a shell.
    subprocess.Popen([exe, str(p)])  # nosec B603
