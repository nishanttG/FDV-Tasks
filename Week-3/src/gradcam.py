import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


class GradCAM:
	def __init__(self, model: torch.nn.Module, target_layer_name: str):
		self.model = model
		self.target_layer_name = target_layer_name
		self.activations = None
		self.gradients = None
		self.hook_handles = []
		self._register_hooks()

	def _find_layer(self):
		parts = self.target_layer_name.split('.')
		module = self.model
		for p in parts:
			module = getattr(module, p)
		return module

	def _register_hooks(self):
		layer = self._find_layer()

		def forward_hook(module, inp, out):
			self.activations = out.detach()

		def backward_hook(module, grad_in, grad_out):
			# grad_out is a tuple
			self.gradients = grad_out[0].detach()

		self.hook_handles.append(layer.register_forward_hook(forward_hook))
		self.hook_handles.append(layer.register_full_backward_hook(backward_hook))

	def remove_hooks(self):
		for h in self.hook_handles:
			h.remove()

	def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
		self.model.zero_grad()
		output = self.model(input_tensor)
		if class_idx is None:
			class_idx = int(output.argmax(dim=1).item())

		score = output[0, class_idx]
		score.backward(retain_graph=True)

		# activations: [B, C, H, W], gradients: [B, C, H, W]
		grads = self.gradients[0]
		acts = self.activations[0]

		weights = grads.mean(dim=(1, 2), keepdim=True)  # global average pooling
		cam = (weights * acts).sum(dim=0)
		cam = F.relu(cam)
		cam = cam - cam.min()
		if cam.max() != 0:
			cam = cam / cam.max()
		cam_np = cam.cpu().numpy()
		return cam_np


def preprocess_image(image_path, device):
	img = Image.open(image_path).convert('RGB')
	transform = transforms.Compose([
		transforms.Resize((32, 32)),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
	])
	t = transform(img).unsqueeze(0).to(device)
	return t, img


def overlay_cam(original_pil: Image.Image, cam_mask: np.ndarray, alpha: float = 0.5):
	heatmap = (255 * cam_mask).astype(np.uint8)
	heat = Image.fromarray(heatmap).resize(original_pil.size).convert('L')
	heat_col = Image.fromarray(np.stack([heatmap, np.zeros_like(heatmap), 255 - heatmap], axis=2)).resize(original_pil.size)
	return Image.blend(original_pil.convert('RGBA'), heat_col.convert('RGBA'), alpha)


if __name__ == '__main__':
	# Minimal CLI to generate a gradcam image when run directly.
	import argparse
	import timm

	parser = argparse.ArgumentParser()
	parser.add_argument('--img', required=True)
	parser.add_argument('--model', required=True)
	parser.add_argument('--layer', default='layer4')
	parser.add_argument('--out', default='gradcam.png')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = timm.create_model('resnet18', pretrained=False, num_classes=10)
	model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
	model.maxpool = torch.nn.Identity()
	model.load_state_dict(torch.load(args.model, map_location=device))
	model.to(device).eval()

	x, orig = preprocess_image(args.img, device)
	cam = GradCAM(model, args.layer)
	mask = cam(x)
	out = overlay_cam(orig, mask)
	out.save(args.out)
	cam.remove_hooks()
