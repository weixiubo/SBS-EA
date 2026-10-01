from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("build_sole2_experiment_matrix.py")),
        run_name="__main__",
    )
