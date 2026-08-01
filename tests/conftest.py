import sys
from pathlib import Path

# Allow running the tests without installing the package (`pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
