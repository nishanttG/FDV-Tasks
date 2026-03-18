import os
import torchvision
from PIL import Image

# 1. Define Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(PARENT_DIR, 'data') 

OUTPUT_DIR = os.path.join(CURRENT_DIR, "images_to_label")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f" Data Source: {DATA_ROOT}")
print(f" Output Dir:  {OUTPUT_DIR}")

# 2. Load CIFAR
try:
    dataset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True)
except:
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# 3. Extract ONLY Planes (0) and Cars (1)
print(f" Extracting Binary Subset (Planes vs Cars)")

count = 0
target_classes = [0, 1] # 0=Plane, 1=Car
class_names = {0: "plane", 1: "car"}

for i in range(len(dataset)):
    img, label = dataset[i]
    
    if label in target_classes:
        # Save the image
        # We include the class name in filename mainly for your reference, 
        # but in a real labeling job, the filename would just be an ID.
        # Let's keep it opaque (ID only) to simulate a real "Unlabeled" task!
        filename = f"img_{count:03d}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        img.save(save_path)
        
        count += 1
        if count >= 200:
            break

print(f" Saved {count} images (Planes & Cars only)")