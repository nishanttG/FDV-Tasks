import os
import pickle
import argparse
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "index")

def load_index():
    with open(os.path.join(INDEX_DIR, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    return bm25, meta["doc_ids"], meta["corpus"]

def search(query, k=5):
    bm25, doc_ids, corpus = load_index()
    
    # Search
    tokenized_query = query.lower().split(" ")
    scores = bm25.get_scores(tokenized_query)
    
    # Get top N
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    results = []
    for idx in top_n:
        results.append({
            "doc_id": doc_ids[idx],
            "score": scores[idx],
            "snippet": corpus[idx][:200] + "..."
        })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini BM25 Search Engine")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    args = parser.parse_args()
    
    start = time.time()
    hits = search(args.query, args.k)
    duration = time.time() - start
    
    print(f"\nSearch Results for: '{args.query}' ({duration:.4f}s)")
    print("-" * 50)
    for i, hit in enumerate(hits):
        print(f"{i+1}. [ID: {hit['doc_id']}] (Score: {hit['score']:.2f})")
        print(f"   Snippet: {hit['snippet']}\n")