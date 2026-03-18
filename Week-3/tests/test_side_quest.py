import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS_CSV = ROOT / "side_quest" / "labels.csv"
SPLITS_CSV = ROOT / "side_quest" / "splits.csv"


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def test_labels_csv_exists():
    assert LABELS_CSV.exists(), f"labels.csv not found at {LABELS_CSV}"


def test_labels_csv_has_header():
    with open(LABELS_CSV, newline='', encoding='utf-8') as f:
        header = f.readline().strip()
    assert header.lower() in ("filename,label", "filename,label\r"), "labels.csv header unexpected"


def test_no_duplicate_filenames_in_labels():
    rows = read_csv(LABELS_CSV)
    filenames = [r['filename'] for r in rows]
    assert len(filenames) == len(set(filenames)), "Duplicate filenames found in labels.csv"


def test_splits_csv_exists():
    assert SPLITS_CSV.exists(), f"splits.csv not found at {SPLITS_CSV}"


def test_splits_cover_same_files_as_labels():
    labels = read_csv(LABELS_CSV)
    splits = read_csv(SPLITS_CSV)
    label_files = set(r['filename'] for r in labels)
    split_files = set(r['filename'] for r in splits)
    assert label_files == split_files, "Mismatch between labels.csv and splits.csv filenames"


def test_splits_ratios_reasonable():
    rows = read_csv(SPLITS_CSV)
    total = len(rows)
    counts = {'train': 0, 'val': 0, 'test': 0}
    for r in rows:
        s = r.get('split', '').lower()
        if s in counts:
            counts[s] += 1

    # Expected 70/15/15 ±10% absolute tolerance
    train_pct = counts['train'] / total
    val_pct = counts['val'] / total
    test_pct = counts['test'] / total

    assert abs(train_pct - 0.70) <= 0.10, f"Train split proportion surprising: {train_pct:.2f}"
    assert abs(val_pct - 0.15) <= 0.10, f"Val split proportion surprising: {val_pct:.2f}"
    assert abs(test_pct - 0.15) <= 0.10, f"Test split proportion surprising: {test_pct:.2f}"
