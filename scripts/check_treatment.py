#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

BASE_DIR = Path.home() / "DATA" / "SLAINTE" / "0.1.5" / "csv" / "plant"
PATTERN = "*_stand_md.csv"
COLUMN = "st_treatment"


def collect_unique_treatments(base_dir: Path):
    treatments = set()

    for csv_file in base_dir.glob(PATTERN):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Skipping {csv_file.name}: read error ({e})")
            continue

        if COLUMN not in df.columns:
            print(f"Skipping {csv_file.name}: column '{COLUMN}' missing")
            continue

        values = (
            df[COLUMN]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )
        treatments.update(values)

    return sorted(treatments)


if __name__ == "__main__":
    treatments = collect_unique_treatments(BASE_DIR)

    print(f"Found {len(treatments)} unique st_treatment values:\n")
    for t in treatments:
        print(f"- {t}")
