import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import timm

from src.dataset import make_dataloaders
from src.utils import seed_everything, save_checkpoint, accuracy, write_metadata, get_device


def build_model(num_classes=10, device=None):
	model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
	# CIFAR patch
	model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
	model.maxpool = nn.Identity()
	if device is not None:
		model.to(device)
	return model


def evaluate(model, loader, device):
	model.eval()
	accs = []
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)
			logits = model(x)
			accs.append(accuracy(logits, y))
	return sum(accs) / len(accs)


def train(args):
	seed_everything(args.seed)
	device = get_device()

	train_loader, val_loader, test_loader = make_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size, val_frac=args.val_frac, num_workers=args.num_workers)

	model = build_model(num_classes=10, device=device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

	if args.scheduler == 'cosine':
		scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	else:
		scheduler = None

	scaler = GradScaler()

	best_val = 0.0
	metadata = {
		'args': vars(args),
		'best_val': None,
	}

	for epoch in range(1, args.epochs + 1):
		t0 = time.time()
		model.train()
		running_loss = 0.0
		for xb, yb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)

			optimizer.zero_grad()
			with autocast(enabled=(device.type == 'cuda')):
				logits = model(xb)
				loss = criterion(logits, yb)

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			running_loss += loss.item() * xb.size(0)

		if scheduler is not None:
			scheduler.step()

		train_loss = running_loss / (len(train_loader.dataset))
		val_acc = evaluate(model, val_loader, device)

		elapsed = time.time() - t0
		print(f"Epoch {epoch}/{args.epochs} — train_loss: {train_loss:.4f} val_acc: {val_acc:.4f} time: {elapsed:.1f}s")

		# Save best
		if val_acc > best_val:
			best_val = val_acc
			ckpt_path = args.output_dir.rstrip('/') + f"/best.pth"
			save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'val_acc': val_acc}, ckpt_path)
			metadata['best_val'] = {'epoch': epoch, 'val_acc': val_acc, 'ckpt': ckpt_path}

	# final test
	test_acc = evaluate(model, test_loader, device)
	metadata['test_acc'] = test_acc
	write_metadata(args.output_dir.rstrip('/') + '/training_metadata.json', metadata)
	print(f"Training complete — best val: {best_val:.4f} test_acc: {test_acc:.4f}")


if __name__ == '__main__':
	p = ArgumentParser()
	p.add_argument('--data-dir', default='data')
	p.add_argument('--batch-size', type=int, default=128)
	p.add_argument('--epochs', type=int, default=10)
	p.add_argument('--lr', type=float, default=0.1)
	p.add_argument('--num-workers', type=int, default=4)
	p.add_argument('--seed', type=int, default=42)
	p.add_argument('--val-frac', type=float, default=0.1)
	p.add_argument('--scheduler', choices=['cosine', 'none'], default='cosine')
	p.add_argument('--output-dir', default='models')
	args = p.parse_args()
	train(args)

