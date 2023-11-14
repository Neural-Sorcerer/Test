import copy
import torch
import matplotlib
import numpy as np

from PIL import Image
from torch.autograd import Variable


class CamExtractor():
	"""
		Extracts cam features from the model
	"""
	def __init__(self, model, target_layer):
		self.model = model
		self.target_layer = target_layer
		self.gradients = None

	def save_gradient(self, grad):
		self.gradients = grad

	def forward_pass_on_convolutions(self, x):
		"""
			Does a forward pass on convolutions, hooks the function at given layer
		"""
		conv_output = None
		for module_pos, module in self.model.features._modules.items():
			x = module(x)  # Forward
			if module_pos == self.target_layer:
				x.register_hook(self.save_gradient)
				conv_output = x  # Save the convolution output on that layer
		return conv_output, x

	def forward_pass(self, x):
		"""
			Does a full forward pass on the model
		"""
		# Forward pass on the convolutions
		conv_output, x = self.forward_pass_on_convolutions(x)
		x = x.view(x.size(0), -1)  # Flatten
		# Forward pass on the classifier
		x = self.model.avgpool(x)			# AvgPool --> Resnet50
		# x = self.model.classifier(x)
		return conv_output, x


# ------------------------GradCam-------------------------
class GradCam():
	"""
		Produces class activation map
	"""
	def __init__(self, model, target_layer):
		device = next(model.parameters()).device

		if device == "cpu":
			self.model = model
		else:
			self.model = copy.deepcopy(model).float().cpu()

		self.model.eval()
		
		# Define extractor
		self.extractor = CamExtractor(self.model, target_layer)

	def generate_cam(self, input_image, target_class=None):
		# Full forward pass
		# conv_output is the output of convolutions at specified layer
		# model_output is the final output of the model (1, 1000)
		conv_output, model_output = self.extractor.forward_pass(input_image)
		if target_class is None:
			target_class = np.argmax(model_output.data.numpy())
		# Target for backprop
		one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
		one_hot_output[0][target_class] = 1
		# Zero grads
		self.model.features.zero_grad()
		self.model.classifier.zero_grad()
		# Backward pass with specified target
		model_output.backward(gradient=one_hot_output, retain_graph=True)
		# Get hooked gradients
		guided_gradients = self.extractor.gradients.data.numpy()[0]
		# Get convolution outputs
		target = conv_output.data.numpy()[0]
		# Get weights from gradients
		weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
		# Create empty numpy array for cam
		cam = np.ones(target.shape[1:], dtype=np.float32)
		# Have a look at issue #11 to check why the above is np.ones and not np.zeros
		# Multiply each weight with its conv output and then, sum
		for i, w in enumerate(weights):
			cam += w * target[i, :, :]
		cam = np.maximum(cam, 0)
		cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
		cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
		cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2], input_image.shape[3]), Image.LANCZOS)) / 255
		return cam
# ------------------------GradCam-------------------------

# ------------------------LayerCAM-------------------------
class LayerCam():
	"""
		Produces class activation map
	"""
	def __init__(self, model, target_layer):
		device = next(model.parameters()).device

		if device == "cpu":
			self.model = model
		else:
			self.model = copy.deepcopy(model).float().cpu()

		self.model.eval()

		# Define extractor
		self.extractor = CamExtractor(self.model, target_layer)

	def generate_cam(self, input_image, target_class=None):
		# Full forward pass
		# conv_output is the output of convolutions at specified layer
		# model_output is the final output of the model (1, 1000)
		conv_output, model_output = self.extractor.forward_pass(input_image)
		if target_class is None:
			target_class = np.argmax(model_output.data.numpy())
		# Target for backprop
		one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
		one_hot_output[0][target_class] = 1
		# Zero grads
		self.model.features.zero_grad()
		self.model.classifier.zero_grad()
		# Backward pass with specified target
		model_output.backward(gradient=one_hot_output, retain_graph=True)
		# Get hooked gradients
		guided_gradients = self.extractor.gradients.data.numpy()[0]
		# Get convolution outputs
		target = conv_output.data.numpy()[0]
		# Get weights from gradients
		weights = guided_gradients
		weights[weights < 0] = 0 # discard negative gradients
		# Element-wise multiply the weight with its conv output and then, sum
		cam = np.sum(weights * target, axis=0)
		cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
		cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
		cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2], input_image.shape[3]), Image.LANCZOS)) / 255
		return cam
# ------------------------LayerCAM-------------------------


def apply_colormap_on_image(org_im, activation, colormap_name):
	"""
		Apply heatmap on image
	Args:
		org_img (PIL img): Original image
		activation_map (numpy arr): Activation map (grayscale) 0-255
		colormap_name (str): Name of the colormap
	"""
	# Get colormap
	color_map = matplotlib.colormaps[colormap_name]
	no_trans_heatmap = color_map(activation)
	# Change alpha channel in colormap to make sure original image is displayed
	heatmap = copy.copy(no_trans_heatmap)
	heatmap[:, :, 3] = 0.4
	heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
	no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

	# Apply heatmap on image
	heatmap_on_image = Image.new("RGBA", org_im.size)
	heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
	heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
	return no_trans_heatmap, heatmap_on_image


def apply_heatmap(frame, ClassActivationMap, coordinates, cropped_frame, cropped_frame_normolised):
	# Get original coordinates
	x0, y0, original_size = coordinates

	# Make a variable from the normalized tensor
	cropped_frame_normolised_variable = Variable(cropped_frame_normolised, requires_grad=True)

	# Generate activation_map (numpy arr): (grayscale) 0-255
	cam = ClassActivationMap.generate_cam(cropped_frame_normolised_variable)

	# Apply heatmap on image
	heatmap, heatmap_on_image = apply_colormap_on_image(cropped_frame, cam, 'hsv')
	
	# Resize the heatmap_on_image
	heatmap_on_image_resized = heatmap_on_image.resize(original_size, Image.LANCZOS)

	# Convert to PIL frame
	frame = Image.fromarray(frame)
	
	# Paste on frame with original coordinates the heatmap_on_image_resized 
	frame.paste(heatmap_on_image_resized, (x0, y0))

	# Convert to numpy arr
	frame = np.array(frame)

	return frame
