"""Run full scenario validation and store a markdown report.

Usage:
    python -m scripts.run_validation --out artifacts/validation_report.md
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from scenarios.runner import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario sweep validation report.")
    parser.add_argument("--out", default="artifacts/validation_report.md", help="Output markdown report path")
    args = parser.parse_args()

    generated = Path(run_all())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if generated.resolve() != out_path.resolve():
        shutil.copyfile(generated, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
