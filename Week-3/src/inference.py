import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import argparse
import os

#  CONFIG 
# 1. CLASSES
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 2. NORMALIZATION (Same as training)
STATS = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

def get_model(device, model_path):
    print(f"Loading model from {model_path}...")
    
    # Re-create exact architecture
    model = timm.create_model('resnet18', pretrained=False, num_classes=10)
    
    #  APPLYING CIFAR PATCH 
    # We must match the training architecture exactly
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    
    model.to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Tip: Ensure you are using the 'ResNet18_FineTune_best.pth' file.")
        exit(1)
        
    model.eval()
    return model

def predict(image_path, model_path, temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = get_model(device, model_path)
    
    # 2. Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # Resize to match CIFAR size
        transforms.ToTensor(),
        transforms.Normalize(*STATS)
    ])
    
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 3. Inference
    with torch.no_grad():
        logits = model(input_tensor)
        
        #  TEMPERATURE SCALING (Calibration) 
        calibrated_logits = logits / temperature
        probabilities = torch.nn.functional.softmax(calibrated_logits, dim=1)
        
        confidence, pred_idx = torch.max(probabilities, 1)
        
    # 4. Output
    pred_class = CLASSES[pred_idx.item()]
    conf_score = confidence.item() * 100
    
    print("-" * 30)
    print(f"  Image:      {image_path}")
    print(f" Prediction: {pred_class.upper()}")
    print(f" Confidence: {conf_score:.2f}% (T={temperature})")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Inference Script")
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for calibration (default: 1.0)")
    args = parser.parse_args()
    
    predict(args.img, args.model, args.temp)