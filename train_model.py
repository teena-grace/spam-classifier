import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ── 1. Load Data ──────────────────────────────────────────
df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])

# ── 2. Preprocess ─────────────────────────────────────────
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})  # ham=0, spam=1

X = df["message"]
y = df["label_num"]

# ── 3. Split Data ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Vectorize Text (Bag of Words) ──────────────────────
vectorizer = CountVectorizer(stop_words="english", lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── 5. Train Naive Bayes Model ────────────────────────────
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ── 6. Evaluate ───────────────────────────────────────────
y_pred = model.predict(X_test_vec)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ── 7. Save Model & Vectorizer ────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model saved!")