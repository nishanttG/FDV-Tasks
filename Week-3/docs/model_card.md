#  Model Card: ResNet18-CIFAR10-FineTuned

## 1. Model Details
*   **Architecture:** ResNet-18 (Adapted for CIFAR-10).
*   **Modifications:** Replaced `Conv1` (7x7->3x3) and removed `MaxPool` to handle 32x32 resolution.
*   **Library:** PyTorch / Timm.
*   **Author:** Nishant Ghimire.

## 2. Intended Use
*   **Task:** Image Classification (10 Classes).
*   **Classes:** Plane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
*   **Input:** RGB Images. Optimal performance on low-resolution inputs. High-res images are automatically resized to 32x32.

## 3. Training Data
*   **Dataset:** CIFAR-10 (50k Train, 10k Test).
*   **Augmentations:** RandomCrop, RandomHorizontalFlip.
*   **Regularization:** Label Smoothing (0.1), Weight Decay (1e-4).

## 4. Performance Metrics
| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Test Accuracy** | **~94.2%** | Fine-tuned for 20 epochs. |
| **Robustness** | **High** | Only ~4% drop at Noise Level 0.1. |
| **Calibration** | **Optimized** | Inference uses Temperature Scaling (T ≈ 0.98). |

## 5. Limitations
*   **Resolution Bias:** The model assumes 32x32 pixel structure. It may struggle with objects that disappear when downsampled (e.g., small text).
*   **Domain:** Strictly for CIFAR-10 classes. Will fail on "Person" or "Bicycle."

## 6. How to Run
```bash
python inference.py --img my_dog.jpg --model ResNet18_FineTune_best.pth --temp 0.9817

