import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph_db import GraphDB
from src.ml_pipeline import ConstitutionClassifier
from src.config import logger

def save_report_text(content, filename="day3_ml_report.txt"):
    report_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    if not os.path.exists(report_dir): os.makedirs(report_dir)
    filepath = os.path.join(report_dir, filename)
    with open(filepath, "w") as f:
        f.write(content)
    logger.info(f"Text Report saved: {filepath}")

def save_confusion_matrix(y_true, y_pred, class_names):
    """Generates and saves Confusion Matrix with tight layout."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Advanced Labels)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    report_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    if not os.path.exists(report_dir): os.makedirs(report_dir)
    filepath = os.path.join(report_dir, "confusion_matrix.png")
    
    # FIX: Use bbox_inches='tight' to prevent cutting off labels
    plt.savefig(filepath, bbox_inches='tight', dpi=120)
    plt.close()
    logger.info(f"Confusion Matrix saved: {filepath}")

def main():
    logger.info("STARTED: Day 3 ML Training (Local w/ Model Save)")

    db = GraphDB()
    pipeline = ConstitutionClassifier(db.driver)
    
    try:
        # 1. Get Data & Label (Advanced)
        df = pipeline.get_advanced_data()
        df = pipeline.label_data(df)
        
        logger.info(f"Loaded {len(df)} articles.")
        print(f"Class Distribution: {df['label'].value_counts().to_dict()}")

        # 2. Train
        X_train, X_test, y_train, y_test, y_pred = pipeline.train(df)

        # 3. Save Model (NEW!)
        model_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'constitution_model.joblib')
        pipeline.save_model(model_path)

        # 4. Generate Reports
        target_names = pipeline.class_names
        
        # A. Console Report
        report_str = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        print("\nCLASSIFICATION REPORT:\n")
        print(report_str)
        
        # B. Save Artifacts
        save_report_text(report_str)
        save_confusion_matrix(y_test, y_pred, target_names)

        logger.info(" Day-3 Complete. Model & Reports saved.")

    except Exception as e:
        logger.error(f" ML Failed: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()