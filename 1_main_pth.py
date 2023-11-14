"""
Description: DriverBehaviourPTH 
	1. Using torchscript model at DMS and OMS positions to predict driver behaviour.
	2. Classes: ["0_default", "1_eating_drinking", "2_calling", "3_smoking"]

Date: 23.10.30
Researcher: Maksym Chernozhukov
"""

import os
import cv2
import torch

from time import time
from pathlib import Path
from datetime import datetime
from collections import deque
from collections import Counter
from torchvision import transforms
from torchvision.transforms import functional as F


SD = (640, 480)
HD = (1280, 720)


class DriverBehaviourPTH:
	def __init__(self, model_path, device='cpu', half=False, top=0, left=0, shift=384, resize=224) -> None:
		self.half = half
		self.device = device
		self.model_path = model_path
		self.classes = ["0_default", "1_eating_drinking", "2_calling", "3_smoking"]
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.transform = transforms.Compose([transforms.ToPILImage(),
											 transforms.Lambda(lambda x: F.crop(x, top=top, left=left, height=shift, width=shift)),
											 transforms.Resize(resize),
											 transforms.ToTensor(), self.normalize,
											 ])

	def load_model(self):
		model = torch.jit.load(self.model_path, map_location=self.device).float()

		for param in model.parameters():
			param.requires_grad_(False)

		if self.half:
			model.half()

		self.model = model.eval()
		return self

	def preprocess(self, frame):
		frame = self.transform(frame)
		frame = frame.unsqueeze(0)

		if self.half: 
			frame = frame.half()

		if self.device == "cuda": 
			frame = frame.to(self.device)
		return frame
	
	def predict(self, frame):
		with torch.no_grad():
			output = self.model(frame)
		return output

	def postprocess(self, model_output, threshold=0.5):
		softmax_output = model_output[0].softmax(dim=0).tolist()
		confidence = max(softmax_output)

		if confidence < threshold:
			result = self.classes[0]
		else:
			result = self.classes[softmax_output.index(confidence)]
		return (result, confidence)


def main(idx=0, freeze=1, threshold=0.5, deque_len=None, record=True):
	""" Main configs to update for testing
	Note: if 'source' is video --> select 'env' and 'cls'

	Args:
		idx (int): Camera index.
		mode (str): Select camera position --> [DMS, OMS]
		freeze (int): Camera reader waitKey. Defaults to 1.
		threshold (float): Classes must have confidence at least above the threshold, otherwise the "0_default" class will be used.
		deque_len (int): Average prediction of last 'N' frames. If "None" then will be selected default options 'bellow'.
		record (bool): Record the inference
	"""
	source = "video"	# [camera, video]
	mode = "DMS"		# [DMS, OMS]
	env = "inside"		# [inside, outside]
	cls = "4_smoking"	# [0_default, 1_eating, 2_drinking, 3_calling, 4_smoking]

	device = 'cpu'		# [cpu, cuda]
	half = False			# [True, False]
	resolution = SD		# [SD, HD-(only for OMS)]

	# Stream capture
	if source == "camera":
		stream = cv2.VideoCapture(idx)
		stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
		stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
	elif source == "video":
		sample_path = f"samples/{mode}/{env}/{cls}"
		samples = sorted(os.listdir(sample_path))
		stream = cv2.VideoCapture(f"{sample_path}/{samples[idx]}")

	# Recording
	if record:
		path = "samples/recorded"
		Path(path).mkdir(parents=True, exist_ok=True)
		cur_date_time = datetime.now().strftime("%d.%m.%Y-%H-%M-%S")
		out = cv2.VideoWriter(f'{path}/{cur_date_time}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution)
	
	# Crop setting
	if mode == "DMS":
		org = (230, 45)
		deque_len = deque_len or 20
		top, left, shift, resize = 55, 135, 370, 256
	elif mode == "OMS":
		org = (50, 150)
		deque_len = deque_len or 40
		if resolution == SD:
			top, left, shift, resize = 30, 330, 260, 256
		elif resolution == HD:
			top, left, shift, resize = 30, 645, 400, 256

	# Model loading
	queue_results = deque(maxlen=deque_len)
	model_path = f"weights/{mode}_resnet50_256_v4_jit_traced.pth"
	model = DriverBehaviourPTH(model_path=model_path, device=device, half=half,
							top=top, left=left, shift=shift, resize=resize).load_model()

	count = 0
	count_skip = 0
	count_warm_up = 5
	while (stream.isOpened):
		count += 1
		start = time()

		success, frame = stream.read()

		if not success:
			break

		if count < count_skip:
			continue

		# Convert to RGB
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Main process
		# ================================================================
		inference_speed_start = time()

		# Get prediction
		model_input = model.preprocess(frame)
		model_output = model.predict(model_input)
		cls, confidence = model.postprocess(model_output, threshold=threshold)

		inference_speed_end = time()
		# ================================================================

		# Get average prediction
		queue_results.append(cls)
		most_frequent_cls = Counter(queue_results).most_common(n=1)[0][0] if queue_results else False

		# Draw results
		cv2.rectangle(frame, (left, top), (left+shift, top+shift), (255, 255, 255), 1, cv2.LINE_AA)
		cv2.putText(frame, f"{most_frequent_cls}", org, cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255), 2, cv2.LINE_AA)

		# Convert to BGR
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

		# Record a video
		if record:
			out.write(frame)

		# Show the frame
		cv2.imshow("Driver Behaviour", frame)

		# WaitKey
		if cv2.waitKey(freeze) & 0xFF == ord('q'):
			break

		# Show inference speed
		print("FPS:", round(1 / (time() - start)))
		print(f"Inference speed FPS: {round(1 / (inference_speed_end - inference_speed_start))}")
		print(f"Inference speed: {round(inference_speed_end - inference_speed_start, 5)} ms")
		print("="*70)

		if count == (count_warm_up+count_skip):
			start_avg = time()

	print("Average FPS:", round(1 / ((time() - start_avg) / (count-count_warm_up-count_skip))))
	
	# Destroy all the windows
	stream.release()
	if record: out.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
