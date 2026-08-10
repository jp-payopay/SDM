from __future__ import annotations

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
        subprocess.Popen(["open", str(p)])
    elif sys.platform.startswith("win"):
        subprocess.Popen(["explorer", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p)])
