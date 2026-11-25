import os
from pathlib import Path

_runtime = Path(__file__).resolve().parents[1] / ".runtime"
_runtime.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_runtime))
os.environ.setdefault("MPLCONFIGDIR", str(_runtime / "matplotlib"))

from .model import NERTYOLO

__all__ = ["NERTYOLO"]
