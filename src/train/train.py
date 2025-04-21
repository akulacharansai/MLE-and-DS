import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/train_processed_reviews.csv"
MODEL_PATH = "outputs/models/model_tfidf.pkl"
VECTORIZER_PATH = "outputs/models/tfidf_vectorizer.pkl"
METRICS_PATH = "outputs/predictions/metrics.txt"

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df["stem_lemmatize_processed"]
y = df["sentiment"].map({'negative': 0, 'positive': 1})

# Vectorize text
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_tfidf, y)

# Evaluate on training itself for now
y_pred = model.predict(X_tfidf)
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)

# Save model and vectorizer
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

# Save metrics
with open(METRICS_PATH, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print(f"Model saved to {MODEL_PATH}")
print(f"Vectorizer saved to {VECTORIZER_PATH}")
print("Metrics written to metrics.txt")
