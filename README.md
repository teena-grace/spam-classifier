# 📧 Email Spam Classifier

A Machine Learning project that classifies emails/SMS as **Spam** or **Not Spam** using Naive Bayes algorithm.

## 🛠 Tools & Technologies
- Python 3.11
- Scikit-learn
- Pandas
- NumPy

## 📁 Project Structure
```
spam-classifier/
├── data/                  # Dataset folder
├── download_data.py       # Downloads the dataset
├── explore.py             # Exploratory data analysis
├── train_model.py         # Trains and saves the model
├── predict.py             # Loads model and predicts
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/spam-classifier.git
cd spam-classifier
```

### 2. Create virtual environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install scikit-learn pandas numpy==1.26.4 scipy==1.11.4 scikit-learn==1.3.2
```

### 4. Download dataset
```bash
python download_data.py
```

### 5. Train the model
```bash
python train_model.py
```

### 6. Run predictions
```bash
python predict.py
```

## 📊 Model Performance
- Algorithm: Multinomial Naive Bayes
- Accuracy: ~98%
- Dataset: UCI SMS Spam Collection (5,574 messages)

## 🧠 Algorithm Flow
```
Email Text → Preprocess → CountVectorizer → Naive Bayes → Spam / Not Spam
```