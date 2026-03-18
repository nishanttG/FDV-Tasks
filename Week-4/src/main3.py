import os
import sys

def main():
    print("Day-3: LoRA Fine-Tuning Summary")
    
    # Define Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "results", "day3")
    
    # 1. Document Config
    config = {
        "Base Model": "Qwen/Qwen2.5-0.5B-Instruct",
        "Method": "PEFT / LoRA",
        "Train Data": "2,000 samples",
        "Eval Data": "500 samples",
        "Test F1": "0.9057",
        "Test Acc": "0.9060",
        "ECE": "0.0308"
    }
    
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"   {k:<15}: {v}")
        
    # 2. Check Artifacts
    print(f"\n Checking Artifacts in: {output_dir}")
    
    # Check for model files in subfolder OR root of day3
    model_path = os.path.join(output_dir, "model")
    if not os.path.exists(os.path.join(model_path, "adapter_model.safetensors")):
        model_path = output_dir # Check parent if subfolder empty

    if os.path.exists(os.path.join(model_path, "adapter_model.safetensors")):
        print("Model Weights found")
        print("\n Success. You can now run: python src/inference.py")
    else:
        print("Model Weights MISSING")

if __name__ == "__main__":
    main()