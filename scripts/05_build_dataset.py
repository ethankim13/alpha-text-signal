"""
Builds the final modeling dataset (drift scores + sector + forward return,
one row per filing) and saves it to data/model_dataset.csv — a small,
inspectable file, unlike the raw HTML cache or the .db file, so it's fine
to commit to git as the actual research dataset.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import dataset_builder


def main():
    df = dataset_builder.build_feature_matrix()

    output_path = Path(__file__).parent.parent / "data" / "model_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()