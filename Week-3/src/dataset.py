import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets


STATS = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


def get_transforms(train: bool = True, aug: str | None = None):
	if train:
		t = [
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(*STATS),
		]
		# Note: advanced augmentations (MixUp/CutMix/RandAugment) are intended to be applied
		# in the training loop or via an external library. Here we keep a simple pipeline.
	else:
		t = [transforms.ToTensor(), transforms.Normalize(*STATS)]

	return transforms.Compose(t)


def make_dataloaders(data_dir: str = "data", batch_size: int = 128, val_frac: float = 0.1, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
	"""Return train, val, test dataloaders for CIFAR-10.

	Args:
		data_dir: path where CIFAR-10 will be downloaded / read from.
		batch_size: batch size for loaders.
		val_frac: fraction of training set to hold out for validation.
		num_workers: dataloader workers.
	"""
	# Train dataset
	train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=get_transforms(train=True))
	test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=get_transforms(train=False))

	# Create train / val split
	n = len(train_ds)
	val_n = int(n * val_frac)
	train_n = n - val_n
	train_subset, val_subset = random_split(train_ds, [train_n, val_n], generator=torch.Generator().manual_seed(42))

	train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

	return train_loader, val_loader, test_loader


if __name__ == "__main__":
	# quick smoke test
	tr, va, te = make_dataloaders(batch_size=16, num_workers=0)
	x, y = next(iter(tr))
	print("Train batch shape:", x.shape, y.shape)
