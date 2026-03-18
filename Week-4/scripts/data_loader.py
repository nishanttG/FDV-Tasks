import os
import pandas as pd
import pandera as pa
from glob import glob
from tqdm import tqdm
from scripts.utils import SEED

def load_aclImdb(data_dir='data/aclImdb'):
    """Reads raw text files from aclImdb folder structure."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at {data_dir}. Please download aclImdb.")

    data = {}
    for split in ['train', 'test']:
        data[split] = []
        for sentiment, label in [('pos', 1), ('neg', 0)]:
            path = os.path.join(data_dir, split, sentiment, '*.txt')
            files = glob(path)
            
            print(f"Loading {split}/{sentiment} ({len(files)} files)...")
            for file_path in tqdm(files, leave=False):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data[split].append({'text': f.read(), 'label': label})
    
    # Convert and Shuffle
    train_df = pd.DataFrame(data['train']).sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = pd.DataFrame(data['test']).sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    return train_df, test_df

def validate_schema(df):
    """Pandera validation to fail fast on bad data."""
    schema = pa.DataFrameSchema({
        "text": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
        "label": pa.Column(int, checks=pa.Check.isin([0, 1]))
    })
    try:
        schema.validate(df)
        return True
    except pa.errors.SchemaError as e:
        print(f" Data Validation Error: {e}")
        raise