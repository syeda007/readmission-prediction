"""
ReadmitAI - Data Preparation from the UCI Download
Builds patient-level train / validation / test splits from the original UCI archive
"Diabetes 130-US Hospitals for Years 1999-2008" (https://archive.ics.uci.edu/dataset/296).

Steps
  1. Read diabetic_data.csv and IDS_mapping.csv from the zip.
  2. Target: readmitted_30d = 1 if readmitted == "<30", else 0.
  3. Split by patient (patient_nbr), never by encounter: 20% of patients for testing, then 20% of the
     remaining patients for validation (about 64 / 16 / 20).
  4. Remove death and hospice discharges (dispositions 11, 13, 14, 19, 20, 21).
  5. Drop `weight` (~97% missing) and four medications with no variation.

Note: the committed files in data/ are the canonical splits used by the trained model. The intermediate
files used to create them were not kept, so this script reproduces the procedure, not the identical patient
assignment. It therefore writes to data/regenerated/ by default and never overwrites the canonical files.

Usage:
    py -3.12 setup_data.py --zip "path/to/diabetes+130-us+hospitals+for+years+1999-2008.zip"
"""

import argparse
import io
import os
import zipfile

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]
DROPPED_COLUMNS = ["weight", "acetohexamide", "examide", "citoglipton", "metformin-pioglitazone"]
OUTPUT_FILES = {
    "training": "readmitai_final_training.csv",
    "validation": "readmitai_final_validation.csv",
    "testing": "readmitai_final_testing.csv",
}


def read_from_zip(archive: zipfile.ZipFile, filename: str) -> bytes:
    matches = [n for n in archive.namelist() if os.path.basename(n) == filename]
    if not matches:
        raise FileNotFoundError(f"{filename} not found in the archive")
    return archive.read(matches[0])


def patient_split(df: pd.DataFrame, test_size: float, seed: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    keep_idx, held_idx = next(splitter.split(df, groups=df["patient_nbr"]))
    return df.iloc[keep_idx], df.iloc[held_idx]


def main():
    parser = argparse.ArgumentParser(description="Create ReadmitAI patient-level splits from the UCI zip.")
    parser.add_argument("--zip", required=True, help="path to the UCI diabetes 130-US hospitals zip")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "data", "regenerated"), help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    if out_dir == os.path.join(BASE_DIR, "data"):
        raise SystemExit("Refusing to overwrite the canonical splits in data/. Choose another --out directory.")
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(args.zip) as archive:
        raw = pd.read_csv(io.BytesIO(read_from_zip(archive, "diabetic_data.csv")), dtype={"diag_1": str, "diag_2": str, "diag_3": str})
        with open(os.path.join(out_dir, "IDS_mapping.csv"), "wb") as f:
            f.write(read_from_zip(archive, "IDS_mapping.csv"))
    print(f"Loaded {len(raw):,} encounters from {raw['patient_nbr'].nunique():,} patients")

    raw["readmitted_30d"] = (raw["readmitted"] == "<30").astype(int)

    development, testing = patient_split(raw, test_size=0.20, seed=args.seed)
    training, validation = patient_split(development, test_size=0.20, seed=args.seed)

    for name, split in [("training", training), ("validation", validation), ("testing", testing)]:
        cleaned = split[~split["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)]
        cleaned = cleaned.drop(columns=[c for c in DROPPED_COLUMNS if c in cleaned.columns])
        path = os.path.join(out_dir, OUTPUT_FILES[name])
        cleaned.to_csv(path, index=False)
        print(
            f"{name:<10} {len(cleaned):>7,} encounters | {cleaned['patient_nbr'].nunique():>6,} patients | "
            f"readmitted {cleaned['readmitted_30d'].mean():.2%} -> {path}"
        )

    overlap = (
        set(training["patient_nbr"]) & set(validation["patient_nbr"]),
        set(training["patient_nbr"]) & set(testing["patient_nbr"]),
        set(validation["patient_nbr"]) & set(testing["patient_nbr"]),
    )
    assert not any(overlap), "patient overlap between splits"
    print("Zero patient overlap verified.")


if __name__ == "__main__":
    main()
