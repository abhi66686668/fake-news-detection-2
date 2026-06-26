# Fake News Detector — AI-Powered News Classification

A Flask web application that detects whether a news article is **REAL** or **FAKE** using **TF-IDF vectorization** and **Logistic Regression**. Paste any news article into the browser, click Analyze, and get an instant prediction.

---

## Features

- **ML Classification** — TF-IDF + Logistic Regression trained on 40,000+ real and fake news articles
- **Instant Prediction** — Paste any article and get a REAL / FAKE result in one click
- **Color-Coded Result** — Green for REAL NEWS, Red for FAKE NEWS
- **Clean UI** — Card-based layout with Font Awesome shield icon and responsive design
- **Pre-trained Model** — `model.pkl` and `vectorizer.pkl` included — no retraining needed to run

---

## Project Structure

```
fake-news-detector/
├── app.py              # Flask app — loads model, handles prediction
├── train.py            # Train TF-IDF + Logistic Regression on dataset
├── requirements.txt    # Python dependencies
├── model/
│   ├── model.pkl       # Trained Logistic Regression model
│   └── vectorizer.pkl  # Fitted TF-IDF vectorizer
├── dataset/            # (not included — see Dataset section below)
│   ├── Fake.csv
│   └── True.csv
├── templates/
│   └── index.html      # Textarea form + prediction result UI
└── static/
    └── style.css       # Card layout styling
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, Flask |
| ML Model | scikit-learn Logistic Regression |
| Text Vectorization | TF-IDF (`TfidfVectorizer`, English stop words removed) |
| Model Storage | joblib |
| Frontend | HTML, CSS, Font Awesome |

---

## How It Works

### Training (`train.py`)
1. Loads `Fake.csv` (label = 0) and `True.csv` (label = 1)
2. Concatenates both into one DataFrame
3. Vectorizes the `text` column using TF-IDF (removes English stop words)
4. Splits 80/20 into train and test sets (`random_state=42`)
5. Trains a Logistic Regression classifier
6. Prints test accuracy and saves `model.pkl` + `vectorizer.pkl`

### Prediction (`app.py`)
1. User pastes news article text into the textarea and submits
2. Flask passes the text through the saved TF-IDF vectorizer
3. Logistic Regression predicts `0` (FAKE) or `1` (REAL)
4. Result displayed as `FAKE NEWS` or `REAL NEWS` on the page

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/BRIJESHTHEPOWER/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app (model already included)

```bash
python app.py
```

Visit `http://localhost:5000` — paste a news article and click **Analyze News**.

---

## Retrain the Model (Optional)

If you want to retrain from scratch:

### Download the Dataset

Get the **ISOT Fake News Dataset** from Kaggle:
[https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place `Fake.csv` and `True.csv` inside the `dataset/` folder, then run:

```bash
python train.py
```

This overwrites `model/model.pkl` and `model/vectorizer.pkl`.

---

## Requirements

```
Flask
pandas
numpy
scikit-learn
joblib
nltk
```

---

## License

This project is open-source and available under the [MIT License](LICENSE).
 
