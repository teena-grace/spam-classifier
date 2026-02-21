import pandas as pd

df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])

print(df.head())
print("\nShape:", df.shape)
print("\nLabel counts:\n", df["label"].value_counts())