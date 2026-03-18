import os
import pickle
import sys
import re
import pandas as pd
from glob import glob
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "aclImdb")
INDEX_DIR = os.path.join(BASE_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def load_corpus():
    print("Loading Corpus from ../data/aclImdb...")
    # We will use a subset (e.g., Test Pos) to keep it "Mini" and fast (~12.5k docs)
    # Using 'test/pos' gives us distinct, high-quality reviews to search
    files = glob(os.path.join(DATA_DIR, "test", "pos", "*.txt"))
    
    docs = []
    doc_ids = []
    
    for f in tqdm(files[:5000]): # Limit to 5000 docs for speed
        doc_id = os.path.basename(f).replace(".txt", "")
        with open(f, 'r', encoding='utf-8') as r:
            text = clean_text(r.read())
            docs.append(text)
            doc_ids.append(doc_id)
            
    return docs, doc_ids

def main():
    # 1. Load Data
    corpus, doc_ids = load_corpus()
    
    # 2. Tokenize
    print("Tokenizing")
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    
    # 3. Build BM25
    print("Building BM25 Index")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 4. Save Artifacts (The "Index")
    print("Saving Index")
    with open(os.path.join(INDEX_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
        
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump({"doc_ids": doc_ids, "corpus": corpus}, f)
        
    print("Indexing Complete")

if __name__ == "__main__":
    main()