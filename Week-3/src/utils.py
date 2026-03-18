import random
import os
import json
import torch
import numpy as np


def seed_everything(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, path: str):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(state, path)


def load_checkpoint(path: str, device='cpu'):
	return torch.load(path, map_location=device)


def accuracy(output: torch.Tensor, target: torch.Tensor):
	with torch.no_grad():
		pred = output.argmax(dim=1)
		correct = pred.eq(target).sum().item()
		return correct / target.size(0)


def write_metadata(path: str, metadata: dict):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(metadata, f, indent=2)


def get_device():
	return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
