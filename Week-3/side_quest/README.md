# Side Quest: Binary Image Labeler — What I did

Summary
- Implemented a small PySide6 desktop labeler (`side_quest/app.py`) and a data preparation script (`side_quest/setup_images.py`) to extract a binary subset (Plane vs Car) from CIFAR-10 into `side_quest/images_to_label/`.
- Performed manual labeling sessions that produced `side_quest/labels.csv` (labels recorded as `filename,label`) and exported `side_quest/splits.csv` (70/15/15 train/val/test split).

Files what they do
- `app.py`: Full PySide6 GUI. Features: open folder picker, display images, keybinds `A`/`B` to save labels to `labels.csv`, progress bar, and an `Export Splits` button that writes `splits.csv` with `split` column.
- `setup_images.py`: Extracts up to 200 images (planes and cars) from the CIFAR-10 training set and writes them to `images_to_label/` as `img_###.jpg` files.
- `images_to_label/`: target folder used by the app. (Populated by `setup_images.py`.)
- `labels.csv`: produced during labeling; contains rows `filename,label`.
- `splits.csv`: produced by `app.py` `Export Splits` action; contains `filename,label,split`.

Findings
- The dataset extraction script creates a reproducible binary subset (Planes vs Cars) and writes ~200 images into `images_to_label/`.
- The labeling app correctly appends labels to `labels.csv` and avoids relabeling images already present in the CSV.
- `Export Splits` implements a deterministic 70/15/15 split (shuffled with `random_state=42`) and writes the `split` column in `splits.csv`.
- The UI requires `PySide6` and `pandas`; the repo includes `requirements.txt` for environment reproducibility.

Quick reproduction steps
1. Create a virtual environment and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Extract images (one-time):

```powershell
python side_quest/setup_images.py
```

3. Launch the app and label images interactively:

```powershell
python side_quest/app.py
```

4. When done labeling, click **Export Splits** in the app — `side_quest/splits.csv` will be generated.