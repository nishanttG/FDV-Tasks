# Day 2: Prompt Engineering & Model Comparison

**Focus:** Benchmarking "Tiny" vs "Medium" LLMs for Sentiment Analysis using various prompting strategies.
**Status:**  Complete
**Notebook:** `notebooks/Day2_Prompt_Engineering.ipynb`


##  Objectives
1.  **Model Showdown:** Compare **Qwen-0.5B** (Tiny, Local-Friendly) vs **Zephyr-7B** (Medium, High-Quality).
2.  **Prompt Strategy:** Evaluate 5 prompt templates (Zero-shot to Chain-of-Thought).
3.  **Ablation Study:** Test Temperature sensitivity (Strict 0.1 vs Creative 0.7).
4.  **Trade-off Analysis:** Quantify the Latency vs. Accuracy trade-off.


##  Methodology

### Models Tested
1.  **Qwen 2.5-0.5B-Instruct:** ~0.5 Billion params. Loaded in float16.
2.  **Zephyr-7B-Beta:** ~7 Billion params. Loaded in 4-bit quantization (to fit T4 GPU).

### Strategies
*   **Zero-Shot Basic / Persona:** Direct classification.
*   **Few-Shot (1-shot / 3-shot):** In-context learning with examples.
*   **Chain of Thought:** Reasoning before answering.


##  Results Summary

###  The Leaderboard (Top Configurations)

| Rank | Model | Strategy | Temp | F1 (Macro) | Accuracy | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1. | **Zephyr-7B** | **Zero-Shot Basic** | **0.7** | **0.9282** | **0.9200** | **3.04s** |
| 2. | Day 1 Baseline | TF-IDF + LogReg | N/A | 0.8800 | 0.8800 | 0.00s |
| 3. | **Qwen-0.5B** | **Zero-Shot Basic** | **0.1** | **0.8667** | **0.8600** | **2.08s** |
| 4. | Zephyr-7B | Zero-Shot Basic | 0.1 | 0.8650 | 0.8600 | 3.47s |
| 5. | Qwen-0.5B | Few-Shot (1-shot) | 0.1 | 0.8572 | 0.8600 | **0.11s**  |


##  Deep Dive & Insights

### 1. The "Zephyr vs. Qwen" Comparison
*   **Accuracy:** Zephyr-7B (0.92 F1) significantly outperformed Qwen-0.5B (0.86 F1). It even beat the Day 1 Baseline (0.88), proving that **larger LLMs can beat supervised models** out of the box.
*   **Speed:** Qwen-0.5B is the clear winner for speed. With 1-shot prompting, it achieved **0.11s latency**, making it ~30x faster than Zephyr (3.4s).

### 2. Prompt Sensitivity (The "Context Trap")
*   **Qwen (Tiny Model):** Struggled with complex prompts.
    *   *Observation:* When given 3 examples (3-shot), F1 dropped to **0.45** (random guessing).
    *   *Theory:* Small models have limited attention capacity; the extra text distracted it rather than helped it.
*   **Zephyr (Medium Model):** Handled context better.
    *   *Observation:* Maintained ~0.78 F1 on 3-shot, though it still performed best with simple Zero-shot instructions.

### 3. Temperature Anomalies
*   **Qwen:** Preferred **Strict (0.1)**. At T0.7, it sometimes outputted lower quality answers.
*   **Zephyr:** Surprisingly peaked at **Creative (0.7)** for the Zero-Shot Basic prompt (0.92 F1). This suggests that for this specific model, a little bit of randomness helped it find the "correct" sentiment more often than strict greedy decoding.

##  Final Verdict

| Scenario | Recommendation |
| :--- | :--- |
| **High Accuracy Required** | Use **Zephyr-7B (Zero-Shot)**. It beats the baseline and provides high-quality labels. |
| **High Speed / Real-Time** | Use **Qwen-0.5B (1-shot)**. It provides decent accuracy (86%) at blazing speeds (0.11s). |


##  Reproducibility
*   **Seed:** 42 (Global)
*   **Dataset:** Stanford IMDb (N=50 Held-out Eval Set)
*   **Logs:** All prompts and responses logged to Weights & Biases.

##  How to Reproduce
1.  Open `notebooks/Day2_Prompt_Engineering.ipynb` in Google Colab (T4 GPU required).
2.  Run the cells 
3.  The script will auto-log results to your W&B account