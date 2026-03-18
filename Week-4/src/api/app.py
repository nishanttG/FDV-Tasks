import sys
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path

#  SMART PATH SETUP 
current_file = Path(__file__).resolve()
project_root = current_file.parent

# Walk up directories until we find 'results' or hit root
while not (project_root / "results").exists():
    if project_root.parent == project_root: 
        raise FileNotFoundError("Could not locate 'results' folder.")
    project_root = project_root.parent

# Add 'scripts' to python path so we can import utils
sys.path.append(str(project_root))
from scripts.utils import load_lora_model, generate_prediction

# Global State 
model_resources = {}

#  Lifespan Manager 
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"API Starting...")
    print(f"Project Root detected at: {project_root}")
    
    # Locate Adapter
    adapter_path_1 = project_root / "results" / "day3" / "model"
    adapter_path_2 = project_root / "results" / "day3"
    
    final_adapter_path = None
    
    if (adapter_path_1 / "adapter_config.json").exists():
        final_adapter_path = adapter_path_1
    elif (adapter_path_2 / "adapter_config.json").exists():
        final_adapter_path = adapter_path_2
        
    if not final_adapter_path:
        print(f"CRITICAL ERROR: Could not find 'adapter_config.json'.")
    else:
        print(f"Loading Adapter from: {final_adapter_path}")
        try:
            model, tokenizer, device = load_lora_model(str(final_adapter_path))
            model_resources["model"] = model
            model_resources["tokenizer"] = tokenizer
            model_resources["device"] = device
            print("Model Loaded & Ready!")
        except Exception as e:
            print(f" Model Load Failed: {e}")

    yield
    print(" API Shutting Down...")
    model_resources.clear()

#  App Definition 
app = FastAPI(title="Sentiment Analysis API", version="2.0", lifespan=lifespan)

#  Data Models 
class ReviewRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    confidence: float #  Added Confidence Field
    latency: float

#  Endpoints 
@app.get("/")
def health_check():
    return {
        "status": "active", 
        "model_loaded": "model" in model_resources,
        "root_dir": str(project_root)
    }

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: ReviewRequest):
    if "model" not in model_resources:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    start_time = time.time()
    try:
        # Unpack the tuple (Label, Score) from the updated utils
        label, score = generate_prediction(
            model_resources["model"], 
            model_resources["tokenizer"], 
            model_resources["device"], 
            request.text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    latency = time.time() - start_time
    
    # Return data matching SentimentResponse model
    return {
        "label": label, 
        "confidence": round(score, 4), 
        "latency": round(latency, 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)