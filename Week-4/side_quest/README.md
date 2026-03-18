#  Side Quest: Mini BM25 Search Engine

A lightweight text search engine built from scratch using `rank_bm25`.

##  Structure
*   `index/`: Stores the pickled BM25 index and metadata.
*   `src/search.py`: CLI tool to query the index.
*   `src/eval.py`: Evaluation script (MRR/nDCG).
*   `qrels.json`: Test set of 10 queries + relevant Doc IDs.

##  How to Run

1.  **Build Index:**
    ```bash
    python src/build_index.py
    ```
2.  **Generate Ground Truth (Qrels):**
    ```bash
    python src/gen_qrels.py
    ```
3.  **Search (CLI):**
    ```bash
    python src/search.py "amazing movie plot" --k 3
    ```
4.  **Evaluate:**
    ```bash
    python src/eval.py
    ```

##  Evaluation Metrics
*   **MRR@5:** Measures how high the relevant result appears.
*   **nDCG@5:** Measures relevance quality.
*   **Target:** >7/10 queries successfully retrieving their source document.