import os
import json
import sys
import warnings

# 1. Suppress annoying Pandera/Pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ensure we can import from 'scripts' regardless of where we run this
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils import set_seed, calculate_md5
from scripts.data_loader import load_aclImdb, validate_schema
from scripts.preprocess import clean_text
from scripts.baseline import BaselineModel

def main():
    # 2. Setup & Config
    set_seed()
    
    # Define directories dynamically
    base_dir = os.path.dirname(os.path.dirname(__file__)) # Goes back to WEEK-4/
    data_dir = os.path.join(base_dir, "data", "aclImdb")
    output_dir = os.path.join(base_dir, "results", "day1")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n 1. House Rules: Reproducibility Checks ")
    print(f" Output Directory: {output_dir}")

    # --- CHECKSUM LOGIC (UPDATED) ---
    # Try finding the tar.gz first, if not, checksum the internal README
    archive_path = os.path.join(base_dir, "data", "aclImdb_v1.tar.gz")
    readme_path = os.path.join(data_dir, "README")
    
    # Check variables to store result
    verified_md5 = "N/A"
    checksum_source = "none"

    if os.path.exists(archive_path):
        verified_md5 = calculate_md5(archive_path)
        print(f"Dataset Archive MD5: {verified_md5}")
        checksum_source = "tar.gz"
    elif os.path.exists(readme_path):
        verified_md5 = calculate_md5(readme_path)
        print(f"Dataset README MD5: {verified_md5}")
        print("   (Verified via extracted folder contents)")
        checksum_source = "extracted_readme"
    else:
        print(" Warning: Could not verify dataset version (No tar.gz or README found).")

    # 3. Load Data
    print("\n 2. Loading Data ")
    try:
        # Use the absolute path 'data_dir' defined above
        train_df, test_df = load_aclImdb(data_dir)
        validate_schema(train_df)
        validate_schema(test_df)
    except Exception as e:
        print(f" Setup failed: {e}")
        return

    # 4. Preprocess
    print("\n 3. Preprocessing ")
    train_df['clean'] = train_df['text'].apply(clean_text)
    test_df['clean'] = test_df['text'].apply(clean_text)

    # 5. Train
    print("\n 4. Training Baseline ")
    model = BaselineModel()
    model.train(train_df['clean'], train_df['label'])

    # 6. Evaluate
    print("\n 5. Evaluation ")
    # Pass the specific day1 output directory
    metrics = model.evaluate(test_df['clean'], test_df['label'], output_dir=output_dir)

    # 7. Save Model & Logs
    model_path = os.path.join(output_dir, "baseline_model.pkl")
    model.save(model_path)
    
    # Log run metadata with correct variables
    run_log = {
        "dataset_checksum_source": checksum_source,
        "dataset_md5": verified_md5,
        "metrics": metrics,
        "seed": 42
    }
    
    run_log_path = os.path.join(output_dir, "run_log.json")
    with open(run_log_path, "w") as f:
        json.dump(run_log, f, indent=4)
        
    print(f"\n Day-1 Complete. Artifacts saved to {output_dir}")

if __name__ == "__main__":
    main()