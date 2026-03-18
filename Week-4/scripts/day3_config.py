"""
Day 3 Configuration & Results Store
"""

# Experiment Configuration
LORA_CONFIG = {
    "Base Model": "Qwen/Qwen2.5-0.5B-Instruct",
    "Method": "PEFT / LoRA",
    "Rank (r)": 16,
    "Alpha": 32,
    "Target Modules": ["q_proj", "v_proj"],
    "Train Data": "2000 samples (aclImdb)",
    "Eval Data": "500 samples (aclImdb)"
}

# Results (Hard-coded from your Colab Run)
FINAL_METRICS = {
    "Final Test Accuracy": "0.9080",
    "Final Test F1": "0.9078",
    "Calibration ECE": "0.0308 (Excellent)"
}

# Artifact Filenames
REQUIRED_FILES = [
    "adapter_model.safetensors", 
    "adapter_config.json"
]