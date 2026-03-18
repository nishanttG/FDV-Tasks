#  Week 4 — NLP & LLMs: Baseline → Prompting → Lightweight Adaptation

**Purpose**
A reproducible project comparing classic NLP baselines (TF‑IDF + Logistic/SVM) with LLM prompting and lightweight adaptation (LoRA/PEFT) on the Stanford ACL IMDB sentiment dataset. Deliverables include baseline code, prompt evaluation, a LoRA adapter, dataset/model cards, and reproducible run manifests.

**Repository Layout**
- `data/` : Raw and processed ACL IMDB files (do not commit large files)
- `notebooks/` : Colab notebooks for Day1/Day2/Day3 experiments
- `src/` : Runnable scripts
  - `src/main1.py` : Day 1 baseline (TF‑IDF + sklearn)
  - `src/main2.py` : Day 2 prompt evaluation (templates + results)
  - `src/main3.py` : Day 3 LoRA flow (saves local artifacts / fallback)
  - `src/inference.py` : Local inference CLI
- `scripts/` : Helpers (data loader, preprocess, utils)
- `results/` : Experiment artifacts (JSON, CSV, plots, adapters)
  - `results/day1/`, `results/day2/`, `results/day3/`
- `notebooks/` : exploratory Colab work (golden source for heavy runs)
- `requirements.in`, `requirements.txt` : pinned dependencies

**Reproducibility (must-haves)**
- **Python:** Use the pinned `requirements.txt` (regenerate with `pip-compile` from `requirements.in`).
- **Seed:** Global seed set via `scripts.utils.set_seed(42)`.
- **Data checksums:** Run `scripts/data_checks.py` (or `scripts.utils.calculate_md5`) before experiments; commit the produced `results/data_checks_*.json`.
- **Artifacts:** Save metrics, plots, prompt logs and PEFT/LoRA adapter folder (`model.save_pretrained("results/day3/model_adapter")`) after Colab runs.
- **Manifests:** Save `run_manifest.json` with Python version, seed, GPU type, hyperparams, and data checksums.

**Quickstart — Local (fast smoke tests)**
1. Create venv & install pinned deps:
   - `pip install pip-tools`
   - `pip-compile requirements.in`
   - `pip install -r requirements.txt`
2. Capture input checks:
   - `python -m scripts.data_checks`  # writes `results/data_checks.json`
3. Run Day 1 baseline:
   - `python src/main1.py`
4. Run Day 2 prompt evaluation (proxy / fast):
   - `python src/main2.py --n-examples 200`
5. Run Day 3 smoke (no GPU required — will fallback if model missing):
   - `python src/main3.py`

**Full experiments (Colab / GPU)**
- Use `notebooks/02_prompts.ipynb` and `notebooks/03_lora.ipynb` in Colab to run the full LLM and LoRA experiments (GPU required).
- In Colab: set `os.environ["WANDB_MODE"] = "offline"` if you do not want to sync to W&B.
- Save the following back to this repo under `results/`:
  - `results/day2/prompt_results.csv` and `results/day2/prompt_summary.json`
  - `results/day3/metrics.json`, `results/day3/plots/*.png`
  - `results/day3/model_adapter/` (PEFT adapter via `save_pretrained`)
  - `results/data_checks_before_run.json` and `results/run_manifest.json`

**What to store in `results/day3/model_adapter/`**
- The adapter folder created with `peft`/`model.save_pretrained(adapter_dir)`.
- A small `tokenizer/` folder only if you modified/resaved tokenization.
- A `manifest.json` with the base model id (e.g., `Qwen/Qwen2.5-0.5B-Instruct`), commit SHA, and training hyperparams.

**Testing & Quality**
- Unit tests: `tests/` should cover data loading, preprocessing, prompt generation, metrics, and inference fallbacks.
- Run tests and coverage:
  - `pytest --maxfail=1 -q`
  - `pytest --cov=./ --cov-report=term-missing -q`  # aim for >=70%

**APIs & Demo**
- Add a small FastAPI app `src/api/app.py` to serve a predict endpoint; use a mock predictor if adapters/base model are not present locally.
- Optional Streamlit UI in `src/app_streamlit.py` for demoing prompts interactively.

**Decision Log & Cards**
- `docs/DATASET_CARD.md` and `docs/MODEL_CARD.md` — include limitations, intended uses, and disallowed uses.