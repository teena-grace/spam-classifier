import pickle

# Load saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_spam(email_text):
    vec = vectorizer.transform([email_text])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]

    label = "🚨 SPAM" if prediction == 1 else "✅ Not Spam"
    confidence = probability[prediction] * 100

    print(f"\nMessage : {email_text}")
    print(f"Result  : {label}")
    print(f"Confidence: {confidence:.2f}%")

# ── Test it ───────────────────────────────────────────────
predict_spam("Congratulations! You've won a free iPhone. Click here to claim now!")
predict_spam("Hey, are we still meeting for lunch tomorrow?")
predict_spam("URGENT: Your bank account has been suspended. Verify now!")
predict_spam("Don't forget to bring the project files to the meeting.")
