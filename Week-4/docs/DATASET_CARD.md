### The Dataset Card (`DATASET_CARD.md`)

#  Dataset Card: Stanford IMDb (aclImdb)

## Summary
The Large Movie Review Dataset (Maas et al., 2011) containing 50,000 highly polar movie reviews.

## Structure
*   **Train:** 25,000 samples (12.5k Positive, 12.5k Negative).
*   **Test:** 25,000 samples (12.5k Positive, 12.5k Negative).
*   **Labeling:** Binary (0 = Negative, 1 = Positive).

##  Preprocessing Pipeline (House Rules)
1.  **Ingestion:** Raw `.txt` files loaded from `aclImdb/train/pos` and `aclImdb/train/neg`.
2.  **Cleaning:** 
    *   HTML tags (`<br />`) replaced with spaces.
    *   Non-alphabetic characters removed (for Baseline).
3.  **Formatting (Day 3):** Converted to ChatML format:
    *   `{"role": "user", "content": "Classify sentiment: [Text]"}`
4.  **Validation:** Schema validated using `Pandera` (String text, Int label).
5.  **Integrity:** MD5 Checksum verified against `aclImdb/README`.