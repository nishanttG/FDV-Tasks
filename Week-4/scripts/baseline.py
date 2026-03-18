import pandas as pd
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scripts.utils import SEED

class BaselineModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(C=1.0, solver='liblinear', random_state=SEED))
        ])

    def train(self, X_train, y_train):
        print("Training Logistic Regression Pipeline...")
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test, output_dir="results/day1"):
        print("Evaluating...")
        y_pred = self.pipeline.predict(X_test)
        
        # Metrics
        metrics = {
            "macro_f1": f1_score(y_test, y_pred, average='macro'),
            "accuracy": accuracy_score(y_test, y_pred)
        }
        
        # Save Metrics
        with open(f"{output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Results: {metrics}")
        
        # Error Buckets
        test_df = pd.DataFrame({'text': X_test, 'label': y_test, 'pred': y_pred})
        fp = test_df[(test_df.label == 0) & (test_df.pred == 1)]
        fn = test_df[(test_df.label == 1) & (test_df.pred == 0)]
        
        fp.head(50).to_csv(f"{output_dir}/false_positives.csv", index=False)
        fn.head(50).to_csv(f"{output_dir}/false_negatives.csv", index=False)
        print(f"Saved {len(fp)} FPs and {len(fn)} FNs to {output_dir}/")
        
        return metrics

    def save(self, filepath):
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")