"""
Loads data/model_dataset.csv (Phase 3a's output), runs every model
through a walk-forward backtest, and prints pooled out-of-sample R^2 and
correlation per model — this is the project's actual finding.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src import modeling


def main():
    data_path = Path(__file__).parent.parent / "data" / "model_dataset.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}\n")

    results = modeling.run_walk_forward_backtest(df)

    print("Walk-forward backtest results (pooled out-of-sample predictions):\n")
    print(f"{'Model':<15} {'R^2':>10} {'Correlation':>12} {'N':>6}")
    for name, r in results.items():
        print(f"{name:<15} {r['r2']:>10.4f} {r['correlation']:>12.4f} {r['n_predictions']:>6}")


if __name__ == "__main__":
    main()