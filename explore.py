from pathlib import Path

import pandas as pd


def load_dataset() -> tuple[pd.DataFrame, Path]:
    data_dir = Path("data")

    candidates = [
        data_dir / "emails.csv",
        data_dir / "emails.csv" / "emails.csv",
        data_dir / "SMSSpamCollection",
    ]

    for path in candidates:
        if path.is_file():
            if path.name == "SMSSpamCollection":
                df = pd.read_csv(path, sep="\t", names=["label", "message"])
            else:
                df = pd.read_csv(path)
            return df, path

    raise FileNotFoundError(
        "No readable dataset found. Expected one of: "
        "data/emails.csv, data/emails.csv/emails.csv, or data/SMSSpamCollection"
    )


df, dataset_path = load_dataset()

print(f"Loaded dataset from: {dataset_path}")
print("\nFirst 5 rows:")
print(df.head())

print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum().sum())

label_candidates = ["label", "Prediction", "prediction", "target"]

for label_column in label_candidates:
    if label_column in df.columns:
        print(f"\nLabel counts ({label_column}):\n", df[label_column].value_counts())
        break
