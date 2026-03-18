import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the shared utils file
from scripts.utils import load_lora_model, generate_prediction

def main():
    print(" Day-3: Local Inference CLI ")
    
    # 1. Locate Adapter
    base_dir = os.path.dirname(os.path.dirname(__file__))
    # Check paths
    adapter_path = os.path.join(base_dir, "results", "day3", "model")
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        adapter_path = os.path.join(base_dir, "results", "day3") # Fallback

    # 2. Load Model
    try:
        model, tokenizer, device = load_lora_model(adapter_path)
        print(" Model Loaded Successfully!")
    except Exception as e:
        print(f" Error loading model: {e}")
        return

    # 3. Interactive Loop
    print("\n Sentiment Analysis Demo (Type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        text = input("\nEnter a movie review: ")
        if text.lower() in ["exit", "quit"]: break
        if not text.strip(): continue
        
        try:
            pred = generate_prediction(model, tokenizer, device, text)
            print(f"Prediction: {pred}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()