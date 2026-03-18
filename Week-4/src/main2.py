import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure we can import from 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We mock the PromptEngine import just to document that it exists in the project
# from scripts.prompt_engine import PromptEngine 

def main():
    print("Day-2: Prompt Engineering & Model Comparison ")
    
    # 1. Setup Directories
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "results", "day2")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Reconstruct FULL Benchmark Data
    # Sourced from your actual Colab Logs (Qwen-0.5B vs Zephyr-7B)
    data = [
        # --- QWEN 0.5B (Temperature 0.1 - Strict) ---
        {"Model": "Qwen-0.5B", "Strategy": "Zero-Shot Basic",   "Temp": 0.1, "F1": 0.8667, "Accuracy": 0.8600, "Latency": 2.08},
        {"Model": "Qwen-0.5B", "Strategy": "Zero-Shot Persona", "Temp": 0.1, "F1": 0.6435, "Accuracy": 0.6800, "Latency": 0.15},
        {"Model": "Qwen-0.5B", "Strategy": "Few-Shot (1-shot)", "Temp": 0.1, "F1": 0.8572, "Accuracy": 0.8600, "Latency": 0.11},
        {"Model": "Qwen-0.5B", "Strategy": "Few-Shot (3-shot)", "Temp": 0.1, "F1": 0.4544, "Accuracy": 0.5600, "Latency": 1.20},
        {"Model": "Qwen-0.5B", "Strategy": "Chain of Thought",  "Temp": 0.1, "F1": 0.6729, "Accuracy": 0.5800, "Latency": 1.83},

        # --- QWEN 0.5B (Temperature 0.7 - Creative) ---
        {"Model": "Qwen-0.5B", "Strategy": "Zero-Shot Basic",   "Temp": 0.7, "F1": 0.8367, "Accuracy": 0.8200, "Latency": 1.70},
        {"Model": "Qwen-0.5B", "Strategy": "Zero-Shot Persona", "Temp": 0.7, "F1": 0.7211, "Accuracy": 0.7400, "Latency": 0.43},
        {"Model": "Qwen-0.5B", "Strategy": "Few-Shot (1-shot)", "Temp": 0.7, "F1": 0.8164, "Accuracy": 0.8200, "Latency": 0.15},
        {"Model": "Qwen-0.5B", "Strategy": "Few-Shot (3-shot)", "Temp": 0.7, "F1": 0.6435, "Accuracy": 0.6800, "Latency": 0.96},
        {"Model": "Qwen-0.5B", "Strategy": "Chain of Thought",  "Temp": 0.7, "F1": 0.7328, "Accuracy": 0.6400, "Latency": 1.88},

        # --- ZEPHYR 7B (Temperature 0.1 - Strict) ---
        {"Model": "Zephyr-7B", "Strategy": "Zero-Shot Basic",   "Temp": 0.1, "F1": 0.8650, "Accuracy": 0.8600, "Latency": 3.47},
        {"Model": "Zephyr-7B", "Strategy": "Zero-Shot Persona", "Temp": 0.1, "F1": 0.8188, "Accuracy": 0.8000, "Latency": 3.34},
        {"Model": "Zephyr-7B", "Strategy": "Few-Shot (1-shot)", "Temp": 0.1, "F1": 0.7749, "Accuracy": 0.7200, "Latency": 3.85},
        {"Model": "Zephyr-7B", "Strategy": "Few-Shot (3-shot)", "Temp": 0.1, "F1": 0.7888, "Accuracy": 0.7800, "Latency": 4.62},
        {"Model": "Zephyr-7B", "Strategy": "Chain of Thought",  "Temp": 0.1, "F1": 0.8006, "Accuracy": 0.7000, "Latency": 3.66},

        # --- ZEPHYR 7B (Temperature 0.7 - Creative) ---
        {"Model": "Zephyr-7B", "Strategy": "Zero-Shot Basic",   "Temp": 0.7, "F1": 0.9282, "Accuracy": 0.9200, "Latency": 3.04},
        {"Model": "Zephyr-7B", "Strategy": "Zero-Shot Persona", "Temp": 0.7, "F1": 0.8387, "Accuracy": 0.8000, "Latency": 3.45},
        {"Model": "Zephyr-7B", "Strategy": "Few-Shot (1-shot)", "Temp": 0.7, "F1": 0.7381, "Accuracy": 0.6400, "Latency": 3.64},
        {"Model": "Zephyr-7B", "Strategy": "Few-Shot (3-shot)", "Temp": 0.7, "F1": 0.7408, "Accuracy": 0.7000, "Latency": 3.77},
        {"Model": "Zephyr-7B", "Strategy": "Chain of Thought",  "Temp": 0.7, "F1": 0.6218, "Accuracy": 0.4800, "Latency": 3.69},
    ]
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "benchmarks.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n FULL Benchmarks ({len(df)} rows) saved to: {csv_path}")

    # 3. Print Leaderboard (Top 10)
    print("\n LEADERBOARD (Top 10 by F1):")
    print("-" * 80)
    print(df.sort_values("F1", ascending=False).head(10)[['Model', 'Strategy', 'Temp', 'F1', 'Accuracy']].to_string(index=False))
    print("-" * 80)

    # 4. Generate Professional Plots
    print("\nGenerating Analysis Plots...")
    
    # Chart A: F1 Comparison (Strict Temp 0.1 only for cleaner view)
    plt.figure(figsize=(12, 6))
    subset = df[df['Temp'] == 0.1]
    sns.barplot(data=subset, x="Strategy", y="F1", hue="Model", palette="viridis")
    plt.title("Impact of Prompt Strategy on F1 Score (Temp=0.1)")
    plt.ylim(0.4, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_strategy_comparison.png"))
    plt.close()

    # Chart B: Latency vs Performance (Includes ALL data points)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x="Latency", 
        y="F1", 
        hue="Model", 
        style="Strategy", 
        size="Temp", # Size indicates Temperature
        sizes=(50, 200),
        alpha=0.8
    )
    plt.title("Trade-off: Latency vs Accuracy (All Experiments)")
    plt.xlabel("Avg Latency (seconds)")
    plt.ylabel("Macro F1 Score")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_tradeoff.png"))
    plt.close()
    
    print(f"Charts saved to {output_dir}/")
    print("\n Day 2 Analysis Complete.")

if __name__ == "__main__":
    main()