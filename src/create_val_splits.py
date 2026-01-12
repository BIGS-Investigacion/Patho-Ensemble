import os
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def create_validation_splits(
    csv_in: str,
    csv_out: str,
    fold_prefix: str = "fold_",
    n_folds: int = 50,
    val_frac: float = 0.15,
    random_state: int = 42
):
    ext = os.path.splitext(csv_in)[1].lower()
    sep = "\t" if ext in {".tsv", ".txt"} else ","

    df = pd.read_csv(csv_in, sep=sep)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_frac,
        random_state=random_state
    )

    for i in range(n_folds):
        col = f"{fold_prefix}{i}"
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in input file.")

        case_label_sets = df.groupby("case_id")[col].agg(lambda vals: set(vals))

        train_cases = case_label_sets[
            case_label_sets.apply(lambda s: s == {"train"})
        ].index.to_numpy()

        if len(train_cases) < 2:
            continue

        case_mut = df.groupby("case_id")[df.columns[2]].first()
        y = case_mut.loc[train_cases].to_numpy()

        _, val_idx = next(splitter.split(train_cases, y))
        val_cases = train_cases[val_idx]

        mask = df["case_id"].isin(val_cases) & (df[col] == "train")
        df.loc[mask, col] = "val"

    df.to_csv(csv_out, sep=sep, index=False)
    print(f"Wrote new file with per-case stratified validation splits to: {csv_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert 15% of 'train' → 'val' per fold at the case_id level."
    )
    parser.add_argument(
        "--input-csv", "-i",
        required=True,
        help="Path to input CSV/TSV file"
    )
    parser.add_argument(
        "--output-csv", "-o",
        required=True,
        help="Path to write output CSV/TSV file"
    )
    parser.add_argument(
        "--fold-prefix",
        default="fold_",
        help="Prefix of fold columns (default: 'fold_')"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=50,
        help="Number of folds (default: 50 → fold_0…fold_49)"
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of train-cases to convert to val (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()
    create_validation_splits(
        csv_in=args.input_csv,
        csv_out=args.output_csv,
        fold_prefix=args.fold_prefix,
        n_folds=args.n_folds,
        val_frac=args.val_frac,
        random_state=args.seed
    )
