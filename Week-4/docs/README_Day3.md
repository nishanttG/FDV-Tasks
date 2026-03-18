#  Day 3: Lightweight Adaptation (LoRA)

**Focus:** Fine-tuning a Small Language Model (SLM) on a modest budget using Low-Rank Adaptation (LoRA).
**Notebook:** `notebooks/Day3_LoRA_Finetuning.ipynb`
**Status:**  Complete

## Objectives
1.  **Efficiency:** Fine-tune `Qwen 2.5-0.5B` using PEFT/LoRA on a T4 GPU.
2.  **Performance:** Surpass the Day 1 Baseline (F1: 0.88) and Day 2 Prompting (F1: 0.86).
3.  **Stretch Booster:** Analyze model confidence using **Calibration Curves** and **ECE** (Expected Calibration Error).

##  Methodology

### 1. Configuration
*   **Model:** Qwen/Qwen2.5-0.5B-Instruct (float16).
*   **Method:** LoRA (Rank `r=16`, Alpha `32`, Dropout `0.05`).
*   **Target Modules:** `q_proj`, `v_proj` (Attention layers).
*   **Train Data:** 10,000 samples (40% of aclImdb train).
*   **Eval Data:** 2,500 held-out samples (10% of aclImdb test).

### 2. Training Dynamics
*   **Time:** ~20 minutes (1 epoch).
*   **Compute:** < 2GB VRAM trainable parameters (0.3% of total model params).
*   **Loss:** Dropped consistently from ~3.5 to ~2.7, indicating successful learning.

##  Results & Analysis

### Final Leaderboard (N=2,500 Gold Standard Eval)

| Metric | Day 1 (TF-IDF) | Day 2 (Prompting) | **Day 3 (LoRA Fine-Tune)** |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.8883 | 0.8600 | **0.9080**  |
| **Macro F1** | 0.8883 | 0.8667 | **0.9078**  |

###  Stretch Booster: Calibration & Reliability
We evaluated how trustworthy the model's confidence scores are.
*   **ECE (Expected Calibration Error):** **0.0308 (3.08%)**
*   **Interpretation:** The model is exceptionally well-calibrated (ECE < 0.05 is SOTA level). When the model predicts a label with 90% confidence, it is correct ~90% of the time.

![Reliability Diagram](https://wandb.ai/nishantg/Week-4?nw=nwusernishanttg) *(See W&B logs for the ECE plot)*

###  Key Insights
1.  **Data Efficiency:** With only 40% of the training data, LoRA Fine-Tuning beat the full-dataset TF-IDF baseline by **~2%**.
2.  **Small Models Win:** A 0.5B parameter model, when fine-tuned, outperforms generic prompting on larger models (Zephyr-7B Zero-shot was ~92% on small N, but Qwen LoRA is consistent at 91% on large N).
3.  **Reliability:** Unlike raw LLMs which can be hallucination-prone, the fine-tuned model shows high reliability (low ECE).

##  Artifacts
*   **W&B Run:** [Link to your Day 3 Big Run]
*   **Confusion Matrix:** Shows balanced performance on Pos/Neg classes.
*   **Script:** `src/day3_finetune.py` (Reproduces logic).