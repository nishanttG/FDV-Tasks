# Week 3 - Deep Learning for Vision: From Scratch -> Transfer -> Attribution

**Overview**
- **Focus:** training engineering, augmentations, transfer learning, calibration, and attribution (Grad‑CAM).
- **Dataset (suggested):** CIFAR-10 — https://www.cs.toronto.edu/~kriz/cifar.html

**Repository layout (important files)**
- `inference.py` — inference and calibrated prediction entrypoint.
- `model_card.md` — model intended use, failure modes, dataset notes.
- `models/` — saved checkpoints (e.g., `ResNet18_FineTune_best.pth`).
- `src/` — training, dataset, gradcam, utilities.
- `side_quest/` — desktop app, `app.py`, `labels.csv`, `splits.csv`.

**Quick setup (recommended)**
1. Create virtual environment and install dependencies:

```bash
conda create -n venv python=3.10 -y
conda activate venv
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt

```

2. Run training notebooks (examples in `notebooks/`) or run training scripts in `src/`.

3. Run the side-quest app (toy example):

```bash
python side_quest/app.py
```

**Reproducibility & best practices**
- Track random seeds (PyTorch, numpy, Python) and log them in training notebooks.
- Save best weights and a training metadata JSON alongside checkpoints (optimizer state, epoch, best metric, command args).
- Use `requirements.txt` to pin package versions used for experiments.

**Where to find things**
- Model checkpoints: `models/`
- Notebooks and experiments: `notebooks/`
- Training code: `src/`
- Side-quest app and labels: `side_quest/`

**Findings & Notes**
- **Data & Augmentations:** implemented train/val/test dataloaders with baseline augmentations (random crop, horizontal flip, normalization). Additional augmentations (MixUp, CutMix, RandAugment) were added and used for ablation studies — see notebooks for per-experiment metrics.
- **Transfer Learning:** fine-tuned a ResNet-18 on CIFAR-10 (checkpoint: `models/ResNet18_FineTune_best.pth`). Reported test accuracy is recorded in the model card: ~94.2%.
- **Calibration & Inference:** applied temperature scaling at inference to improve calibration (see `docs/model_card.md`). Inference entrypoint: `inference.py`.
- **Attribution:** generated Grad‑CAM visualizations for misclassified and correctly classified examples (see `notebooks/` and `src/gradcam.py`). These help localize salient regions used by the model.
- **Reproducibility:** best checkpoints and a metadata note are saved in `models/`. Use `requirements.txt` to recreate the environment.

**How I ran experiments (quick commands)**
- Run the main notebook (recommended): open [notebooks/Week_3_Day4_final.ipynb](notebooks/Week_3_Day4_final.ipynb#L1) and run cells top-to-bottom.
- Inference (single image):

```bash
python inference.py --img path/to/image.jpg --model models/ResNet18_FineTune_best.pth --temp 0.9817
```

- Generate Grad‑CAM for an image (script):

```bash
python src/gradcam.py --img path/to/image.jpg --model models/ResNet18_FineTune_best.pth
```

- Run the side-quest labeling app (prototype):

```bash
python side_quest/app.py
```

**Where to look for results**
- Training logs and figures: `notebooks/` (training curve cells) and any `runs/` or `logs/` folders created during training.
- Best model checkpoint: `models/ResNet18_FineTune_best.pth`.
- Documentation and model card: [docs/model_card.md](docs/model_card.md#L1).

Quick usage examples
- Single-image inference:
```powershell
python inference.py --img path/to/image.jpg --model models/ResNet18_FineTune_best.pth --temp 0.9817
```

- Run the main notebook (recommended): open [notebooks/Week_3_Day4_final.ipynb](notebooks/Week_3_Day4_final.ipynb#L1) and run cells top-to-bottom.

