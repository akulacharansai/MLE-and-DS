#  Sentiment Classification using Logistic Regression & TF-IDF

This project focuses on binary sentiment classification of movie reviews using Natural Language Processing (NLP) techniques and machine learning, built with reproducibility in mind via Docker.

---

##  Data Science Part (DS Report)

###  Objective
To classify movie reviews as either **positive** or **negative** using logistic regression and efficient text vectorization methods.

###  Text Preprocessing
- Lowercasing, punctuation and digit removal
- Stop word removal
- Tokenization
- Lemmatization & Stemming

###  Feature Engineering
- **TF-IDF Vectorization**: Captures importance of words across the corpus
- Comparison made with CountVectorizer, but TF-IDF yielded better results.
Outputs are saved under outputs/figures section. Which include confussion matrix and precision-recall curve representation.
###  Model Choice
- **Logistic Regression** was selected for its simplicity, performance on small-to-medium datasets, and interpretability.

###  Performance
Final model achieved:
- **Accuracy**: `89.21%`
*(See `outputs/predictions/metrics.txt` after inference for detailed classification report.)*

---

##  Machine Learning Engineering (MLE Part)



##  Setup & Quickstart (With Docker)

###  Prerequisites
- Docker installed and running
- Place your CSV files:
  - `train_processed_reviews.csv` in `data/raw/`
  - `test_processed_reviews.csv` in `data/raw/`

---

###  1. Train the Model

1. **Build Docker image**
   ```bash
   docker build -t sentiment-train -f src/train/Dockerfile .
2. **Run Training**
docker run --rm -v "$PWD:/app" sentiment-train

###  2. Run Inference

1. **Build Docker image**
    docker build -t sentiment-infer -f src/inference/Dockerfile .
2. **Run Inference**
    docker run --rm -v "$PWD:/app" sentiment-infer


#### Dependencies are listed in requirements.txt


### Reproducibility & Best Metric
-- The F1-score stored in outputs/predictions/metrics.txt is the best metric from testing.
-- Running run_inference.py in Docker reproduces this result automatically