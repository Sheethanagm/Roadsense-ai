from __future__ import annotations

import subprocess
import sys
from pathlib import Path


APP_PATH = Path(__file__).resolve().parent / "app.py"

raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP_PATH)]))
