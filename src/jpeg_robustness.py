"""Deprecated alias for the JPEG axis of the robustness suite (Task 6 -> Task 12).

Use ``python -m src.robustness --perturbations jpeg ...`` directly. This shim
forwards to it so older commands keep working.
"""

from __future__ import annotations

import sys

from .robustness import main as _robustness_main

__all__ = ["main"]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--perturbations" not in argv:
        argv += ["--perturbations", "jpeg"]
    print("NOTE: src.jpeg_robustness is deprecated — use src.robustness", flush=True)
    return _robustness_main(argv)


if __name__ == "__main__":
    main()
