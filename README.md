# IMDB Sentiment Analysis using NLP & Machine Learning

This project analyzes movie reviews from the IMDB dataset and classifies them as **Positive** or **Negative** using Natural Language Processing (NLP) techniques and a machine learning model.

---

## Project Overview

- **Goal**: Build a sentiment classification model to predict the sentiment of a movie review.
- **Tech Stack**:
  - Python
  - NLTK, spaCy
  - Scikit-learn
  - Pandas, NumPy
  - Matplotlib, Seaborn

---

## Dataset

- Input CSV file: `dataset.csv`
- Contains 50,000 IMDB movie reviews with labels:
  - `review`: Text of the movie review.
  - `sentiment`: `positive` or `negative`

---

##  Project Pipeline

### 1. **Preprocessing**
- Remove HTML tags, punctuation, and numbers
- Convert to lowercase
- Tokenize using `nltk`
- Remove stopwords
- Lemmatize using `spaCy`

### 2. **Train/Test Split**
- Stratified 80/20 split using `train_test_split`

### 3. **Feature Extraction**
- TF-IDF Vectorization with 5,000 most frequent terms

### 4. **Model Training**
- Logistic Regression using `liblinear` solver

### 5. **Evaluation**
- Accuracy score
- Classification report
- Confusion matrix visualization

---

## 🧪 Example Predictions

```python
predict_sentiment("This movie was absolutely fantastic!")  # Positive
predict_sentiment("It was a complete waste of time.")     # Negative
predict_sentiment("It was okay, not great but not bad.")  # Negative
