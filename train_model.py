from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def load_dataset() -> tuple[pd.DataFrame, Path]:
    data_dir = Path("data")
    candidates = [
        data_dir / "emails.csv",
        data_dir / "emails.csv" / "emails.csv",
    ]

    for path in candidates:
        if path.is_file():
            return pd.read_csv(path), path

    raise FileNotFoundError(
        "No readable dataset found. Expected data/emails.csv or data/emails.csv/emails.csv"
    )


df, dataset_path = load_dataset()
print(f"Loaded dataset from: {dataset_path}")
print("Dataset shape:", df.shape)

label_column = None
for candidate in ["Prediction", "prediction", "label", "target"]:
    if candidate in df.columns:
        label_column = candidate
        break

if label_column is None:
    raise ValueError("Could not find a label column in the dataset.")

df = df.drop(columns=["Email No."], errors="ignore")
X = df.drop(columns=[label_column])
y = df[label_column]

print("Feature shape:", X.shape)
print(f"Label column: {label_column}")
print("Label counts:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("features.pkl", "wb") as features_file:
    pickle.dump(X.columns.tolist(), features_file)

print("\nModel and features saved.")
