import os
import zipfile
import pandas as pd

def setup_data():
    base_dir = r"d:\ads-project\ReadmitAI"
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "src"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "reports", "figures"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "reports", "metrics"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "api"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "tests"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "static"), exist_ok=True)

    # Extract IDS_mapping.csv
    zip_path = r"C:\Users\DELL\Downloads\diabetes+130-us+hospitals+for+years+1999-2008.zip"
    with zipfile.ZipFile(zip_path) as z:
        z.extract("IDS_mapping.csv", os.path.join(base_dir, "data"))

    # Process the three splits into readmitai_final_*.csv
    excl_disch = [11, 13, 14, 19, 20, 21]
    cols_drop = ["acetohexamide", "examide", "citoglipton", "metformin-pioglitazone"]

    splits = [
        ("readmitai_train.csv", "readmitai_final_training.csv"),
        ("readmitai_validation.csv", "readmitai_final_validation.csv"),
        ("readmitai_test.csv", "readmitai_final_testing.csv"),
    ]

    for src_name, dst_name in splits:
        src_path = os.path.join(r"d:\ads-project\temp_extracted", src_name)
        df = pd.read_csv(src_path)
        # Cohort exclusion:
        df_clean = df[~df["discharge_disposition_id"].isin(excl_disch)].copy()
        # Drop constant features:
        df_clean = df_clean.drop(columns=cols_drop)
        dst_path = os.path.join(base_dir, "data", dst_name)
        df_clean.to_csv(dst_path, index=False)
        rate = df_clean["readmitted_30d"].mean() * 100
        print(f"Created {dst_name}: shape={df_clean.shape}, readmit_rate={rate:.2f}%")

    print("Data setup completed successfully.")

if __name__ == "__main__":
    setup_data()
