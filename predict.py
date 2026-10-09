from pathlib import Path
import pickle

import pandas as pd


MODEL_PATH = Path("model.pkl")
FEATURES_PATH = Path("features.pkl")

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    missing = []
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH))
    if not FEATURES_PATH.exists():
        missing.append(str(FEATURES_PATH))
    raise FileNotFoundError(
        f"Missing required file(s): {', '.join(missing)}. "
        "Run 'python train_model.py' first."
    )

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

with FEATURES_PATH.open("rb") as features_file:
    feature_columns = pickle.load(features_file)

print(f"Model loaded | Vocabulary size: {len(feature_columns)} words\n")


def predict_spam(email_text: str) -> None:
    email_words = email_text.lower().split()
    row = dict.fromkeys(feature_columns, 0)

    for word in email_words:
        cleaned_word = word.strip(".,!?;:\"'()[]{}")
        if cleaned_word in row:
            row[cleaned_word] += 1

    input_df = pd.DataFrame([row], columns=feature_columns)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    confidence = probability[prediction] * 100
    label = "SPAM" if prediction == 1 else "Not Spam"

    print(f"Message    : {email_text[:80]}{'...' if len(email_text) > 80 else ''}")
    print(f"Result     : {label}")
    print(f"Confidence : {confidence:.2f}%")
    print("-" * 60)


predict_spam("Congratulations! You won a free prize. Click here to claim your reward now!")
predict_spam("Hey, are we still on for the meeting tomorrow at 10am?")
predict_spam("URGENT! Your account will be suspended. Verify your details immediately!")
predict_spam("Please find attached the project report for your review.")
predict_spam("Win cash money free offer limited time click now winner selected!")
predict_spam("Can you send me the notes from today's lecture?")
predict_spam("Get free meds delivered to your door. No prescription needed. Order now!")
predict_spam("Win a cash prize! Click here to claim your reward. Limited time offer!")
