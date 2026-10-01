#!/usr/bin/env python3
"""Compatibility launcher using this project's selected Python environment."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
if not sys.argv[1:] or sys.argv[1] == "--info":
    print(json.dumps({"cwd": str(ROOT), "python": sys.executable,
                      "usage": "python 运行.py <script or -m module>"}, ensure_ascii=False))
else:
    raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]], cwd=ROOT))
