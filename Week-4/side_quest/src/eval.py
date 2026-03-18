import json
import os
import numpy as np
from search import search

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def calculate_mrr(rank):
    """Reciprocal Rank: 1/rank. If not found, 0."""
    if rank is None: return 0.0
    return 1.0 / rank

def calculate_ndcg(rank, k=5):
    """Simple binary nDCG. If relevant doc is at 'rank', Gain=1."""
    if rank is None or rank > k: return 0.0
    return 1.0 / np.log2(rank + 1) # Rank 1 -> log2(2)=1 -> Score 1. Rank 2 -> 0.63...

def main():
    # Load Qrels
    with open(os.path.join(BASE_DIR, "qrels.json"), "r") as f:
        qrels = json.load(f)
    
    mrr_scores = []
    ndcg_scores = []
    hits = 0
    
    print(f" Evaluating {len(qrels)} queries...")
    print("-" * 60)
    print(f"{'Query ID':<5} | {'Rank':<5} | {'MRR':<5} | {'Snippet'}")
    print("-" * 60)
    
    for i, item in enumerate(qrels):
        query = item['query']
        targets = item['relevant_docs']
        
        # Search (k=5)
        results = search(query, k=5)
        
        # Check if any target doc is in results
        found_rank = None
        for rank, res in enumerate(results, 1):
            if res['doc_id'] in targets:
                found_rank = rank
                break
        
        # Metrics
        mrr = calculate_mrr(found_rank)
        ndcg = calculate_ndcg(found_rank)
        
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
        
        if found_rank: hits += 1
        
        snippet = results[0]['snippet'][:30] if results else "No results"
        print(f"{i:<8} | {str(found_rank):<5} | {mrr:.2f}  | {snippet}...")

    print("-" * 60)
    print(f"Final Results:")
    print(f"MRR@5:  {np.mean(mrr_scores):.4f}")
    print(f"nDCG@5: {np.mean(ndcg_scores):.4f}")
    print(f"Recall: {hits}/{len(qrels)} queries found relevant hit.")

if __name__ == "__main__":
    main()