import os
import random
import numpy as np
import hashlib
import json
from pathlib import Path

# --- PART 1: GENERAL UTILS ---
SEED = 42

def set_seed(seed=SEED):
    """Locks all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def calculate_md5(filepath):
    """Calculates MD5 checksum."""
    if not os.path.exists(filepath):
        return None
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# --- PART 2: MODEL UTILS ---

def load_lora_model(adapter_path):
    """Loads Base Qwen model + LoRA adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Check Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  Inference Device: {device}")

    # Load Tokenizer
    print(f"⏳ Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Load Base Model
    print(f"⏳ Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float32, 
        device_map=device
    )

    # Load Adapter
    print(f"🔗 Attaching LoRA Adapter from: {adapter_path}")
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        raise FileNotFoundError(f"Adapter config not found at {adapter_path}")

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer, device

def generate_prediction(model, tokenizer, device, text):
    """
    Returns (Label, Confidence Score).
    Uses Softmax on the logits of the specific tokens.
    """
    import torch
    import numpy as np
    
    # 1. Format Input
    messages = [{"role": "user", "content": f"Classify sentiment:\n{text}"}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    
    # 2. Forward Pass (Calculate Probabilities)
    with torch.no_grad():
        outputs = model(inputs)
        # Get logits of the last token
        next_token_logits = outputs.logits[0, -1, :]
        
        # 3. Get Scores for "Positive" vs "Negative"
        # We find the token IDs specifically for these words
        pos_id = tokenizer.encode("Positive", add_special_tokens=False)[0]
        neg_id = tokenizer.encode("Negative", add_special_tokens=False)[0]
        
        pos_score = next_token_logits[pos_id].item()
        neg_score = next_token_logits[neg_id].item()
        
        # 4. Softmax Calculation
        # prob_positive = e^pos / (e^pos + e^neg)
        confidence_pos = np.exp(pos_score) / (np.exp(pos_score) + np.exp(neg_score))
        confidence_neg = 1.0 - confidence_pos
        
    # 5. Determine Label & Confidence
    if confidence_pos > 0.5:
        return "Positive", float(confidence_pos)
    else:
        return "Negative", float(confidence_neg)