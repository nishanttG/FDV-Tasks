# NLP Internship — Week 4: Sentiment Analysis Baseline

This repository implements a reproducible NLP baseline for sentiment analysis on the Stanford IMDb Movie Review dataset using a TF–IDF feature extractor and a Logistic Regression classifier.

**Key goals:** reproducibility, clear tracking of metrics/artifacts, and a simple baseline for comparisons.

## House rules / reproducibility

- **Random seed:** fixed to 42 for deterministic behaviour.
- **Validation:** input schemas enforced via Pandera.
- **Integrity:** MD5 checksums of raw inputs are recorded.
- **Tracking:** metrics (Macro-F1, Accuracy) and error buckets (false positives / false negatives) are saved to `results/`.

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Prepare the dataset

- Download the Stanford Large Movie Review dataset (ACL IMDB) and extract it into `data/aclImdb/`.
- Ensure the `train/` and `test/` folders are present under `data/aclImdb/`.

3. Run the day-1 pipeline

```bash
python main1.py
```

Note: the pipeline will produce model metrics and artifact CSVs under `results/day1/`.

## Expected Results (Day 1)

- Macro F1: ~0.88
- Accuracy: ~0.88

## Artifacts

- `results/day1/metrics.json` — aggregated metrics
- `results/day1/false_positives.csv` — examples the model labeled positive but are negative
- `results/day1/false_negatives.csv` — examples the model labeled negative but are positive
