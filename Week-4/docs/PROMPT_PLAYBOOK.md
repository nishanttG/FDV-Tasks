# Prompt Engineering Playbook

**Goal:** Guidelines and record of prompts tried for the sentiment classification tasks (Qwen/Zephyr and similar models), plus a recommended prompt with usage notes.

## Recommended Strategy
Use concise, few-shot prompts (1–3 examples) with a clear output format. Where label ambiguity exists, include label definitions and one example of each label.


## Prompts Tried (from `02_prompts.ipynb`)

- **Zero-Shot Basic**

  ```text
  Classify the sentiment of this movie review as "Positive" or "Negative"
  Review: "{INPUT_TEXT}"
  Sentiment:
  ```

- **Zero-Shot Persona**

  ```text
  You are an expert film critic. Analyze the following review.
  If the reviewer liked the movie, say "Positive". If they disliked it, say "Negative".
  Review: "{INPUT_TEXT}"
  Sentiment:
  ```

- **Few-Shot (1-shot)**

  ```text
  Task: Classify Movie Reviews.
  Example:
  Review: "{EXAMPLE_TEXT_TRUNCATED}..."
  Sentiment: {EXAMPLE_LABEL}

  Review: "{INPUT_TEXT}"
  Sentiment:
  ```

- **Few-Shot (3-shot)**

  ```text
  Task: Classify Movie Reviews.
  Review: "{EX1_TEXT_TRUNCATED}..."
  Sentiment: {EX1_LABEL}
  Review: "{EX2_TEXT_TRUNCATED}..."
  Sentiment: {EX2_LABEL}
  Review: "{EX3_TEXT_TRUNCATED}..."
  Sentiment: {EX3_LABEL}

  Review: "{INPUT_TEXT}"
  Sentiment:
  ```

- **Chain of Thought**

  ```text
  Analyze the movie review.
  Step 1: Identify emotional keywords.
  Step 2: Decide if the tone is Positive or Negative.
  Format:
  Reasoning: [Reasoning]
  Sentiment: [Label]

  Review: "{INPUT_TEXT}"
  ```


## Quick comparison (practical takeaways)

- **Zero-shot**: Fast, cheap, lower accuracy on nuanced cases.
- **1–3 shot few-shot**: Best practical tradeoff — use 1 shot for short prompts, 3 shots when labels need clarity.
- **COT**: Use sparingly for difficult examples; be mindful of cost.


## Recommended Prompt (empirical)

Summary of findings across notebooks:
- The notebooks evaluated strategies: `Zero-Shot Basic`, `Zero-Shot Persona`, `Few-Shot (1-shot)`, `Few-Shot (3-shot)`, and `Chain of Thought` across temperatures (example: 0.1 and 0.7).
- Experiment logs (W&B summaries in `02_prompts.ipynb`) show strong performance for `Zero-Shot Basic` at higher temperature (e.g., T=0.7) and consistently good, more conservative performance for `Few-Shot (1-shot)` at low temperature (T=0.1).
- `Chain of Thought` delivered mixed gains at higher latency and inconsistent results across models/temps.

Recommendation (based on experiments):
- Default: **Zero-Shot Basic** with temperature ≈ **0.7** — best single-run accuracy in the notebook summaries and simple to use in pipelines.
- Conservative/consistent alternative: **Few-Shot (1-shot)** with temperature ≈ **0.1** — slightly lower variance and good for conservative labeling.

Recommended prompt templates (use exactly):

- Zero-Shot Basic (recommended default):

```text
Classify the sentiment of this movie review as "Positive" or "Negative"
Review: "{INPUT_TEXT}"
Sentiment:
```

- Few-Shot (1-shot) (conservative alternative):

```text
Task: Classify Movie Reviews.
Example:
Review: "{EXAMPLE_TEXT_TRUNCATED}..."
Sentiment: {EXAMPLE_LABEL}

Review: "{INPUT_TEXT}"
Sentiment:
```

Notes & trade-offs:
- Use the Zero-Shot Basic prompt for fast, high-accuracy runs when you can tune temperature (T≈0.7 performed best in our logs).
- Use Few-Shot (1-shot) + low temperature (T≈0.1) when you want more conservative, repeatable outputs.
- Use Chain-of-Thought only for targeted, difficult examples; it increases latency and token cost and showed variable gains.
