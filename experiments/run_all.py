"""Run all three numerical experiments in dependency order.

Usage (from the project root):
    .venv\\Scripts\\python.exe experiments\\run_all.py [--rebuild]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp1_certificate_validation  # noqa: E402
import exp2_wee_robustness_landscape  # noqa: E402
import exp3_selection_sensitivity  # noqa: E402


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    t0 = time.time()
    print("#################### Experiment 2 ####################")
    exp2_wee_robustness_landscape.main(rebuild=rebuild)
    print("\n#################### Experiment 1 ####################")
    exp1_certificate_validation.main()
    print("\n#################### Experiment 3 ####################")
    exp3_selection_sensitivity.main()
    print(f"\nAll experiments finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
