# Mailguard — Email Spam Classifier

A glassmorphism web interface for the existing Scikit-learn email classifier. Paste an email to get a spam likelihood, confidence score, and a short safety explanation. Analysis runs locally through Flask; the browser keeps recent results in session storage.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies and start the app:

```bash
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The app loads `model.pkl` and `features.pkl` from the project root. If those files are missing, run `python train_model.py` first. The training script currently uses `data/emails.csv/emails.csv`.

## Project structure

```text
app.py                 Flask web app and prediction API
templates/index.html   Website interface
static/style.css       Orange and white glass UI
static/app.js          Input, result, examples, and session history
train_model.py         Train the Multinomial Naive Bayes model
predict.py             Command-line prediction examples
model.pkl              Trained model
features.pkl           Model vocabulary
data/                  Training datasets
```

## Notes

- The model was trained on a word-count email dataset and uses its learned vocabulary for classification.
- A “likely safe” result is a model prediction, not a guarantee. Treat unexpected links and requests carefully.
- Recent analysis history stays in the current browser session and clears when that session ends (or when you clear it).
