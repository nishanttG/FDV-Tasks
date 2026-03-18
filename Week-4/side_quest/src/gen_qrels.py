import os
import pickle
import json
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "index")

def generate_qrels():
    # Load data
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    
    corpus = meta["corpus"]
    doc_ids = meta["doc_ids"]
    
    qrels = []
    
    # Pick 10 random documents to be our "Relevant" answers
    indices = random.sample(range(len(corpus)), 10)
    
    for idx in indices:
        text = corpus[idx]
        doc_id = doc_ids[idx]
        
        # Create a query by taking 3-5 distinctive words from the middle
        words = text.split()
        if len(words) > 20:
            start = random.randint(0, len(words)-10)
            query_words = words[start:start+5]
            query = " ".join(query_words)
        else:
            query = text[:30]
            
        qrels.append({
            "query": query,
            "relevant_docs": [doc_id] # The doc we took the words from is definitely relevant!
        })
    
    output_path = os.path.join(BASE_DIR, "qrels.json")
    with open(output_path, "w") as f:
        json.dump(qrels, f, indent=4)
        
    print(f" Generated 10 test queries in {output_path}")

if __name__ == "__main__":
    generate_qrels()    