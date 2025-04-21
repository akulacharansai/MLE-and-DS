import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = "outputs/models/model_tfidf.pkl"
VECTORIZER_PATH = "outputs/models/tfidf_vectorizer.pkl"
TEST_PATH = "data/processed/test_processed_reviews.csv"
OUTPUT_PRED_PATH = "outputs/predictions/predictions.csv"
METRICS_PATH = "outputs/predictions/metrics.txt"

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Load test data
df_test = pd.read_csv(TEST_PATH)
X_test = df_test["stem_lemmatize_processed"]
y_test = df_test["sentiment"].map({'negative': 0, 'positive': 1})

# Transform and predict
X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)

# Save predictions
df_test["predicted_sentiment"] = y_pred
df_test.to_csv(OUTPUT_PRED_PATH, index=False)

# Save metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

with open(METRICS_PATH, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("Inference complete. Predictions and metrics saved.")
