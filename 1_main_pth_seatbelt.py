"""
Description: SeatbeltPTH 
	1. Using torchscript model at OMS-Anchor position to predict Seat belts ON/OFF.
	2. Classes: ["0_off", "1_on"]

Date: 2023.11.03
Researcher: Maksym Chernozhukov
"""
# @ second updatye 2e2ed2e 2edasdas
import os
import cv2
import copy
import torch

from time import time
from pathlib import Path
from datetime import datetime
from collections import deque
from collections import Counter
from torchvision import transforms
from torchvision.transforms import functional as F


SS = (640, 360)
HD = (1280, 720)


class AddPadding():
	def __init__(self, fill=0, padding_mode='constant'):
		self.fill = fill
		self.padding_mode = padding_mode

	def __call__(self, img):
		width, height = img.size

		if width != height:
			pad = int((height - width) // 2)
			img = transforms.Pad(padding=(pad, 0, pad, 0), fill=self.fill, padding_mode=self.padding_mode)(img)
		return img


class SeatbeltPTH:
	def __init__(self, model_path, device='cpu', half=False, resize=256) -> None:
		self.half = half
		self.device = device
		self.model_path = model_path
		self.classes = ["0_off", "1_on"]
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.transform = transforms.Compose([transforms.ToPILImage(),
											 AddPadding(fill=0, padding_mode='constant'),
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


def main(idx=4, freeze=1, threshold=0.5, deque_len=20, record=False, resolution=SS):
	""" Main configs to update for testing
	Note: if 'source' is video --> select 'cls'

	Args:
		idx (int): Camera index.
		freeze (int): Camera reader waitKey. Defaults to 1.
		threshold (float): Classes must have confidence at least above the threshold, otherwise the "0_default" class will be used.
		deque_len (int): Average prediction of last 'N' frames. If "None" then will be selected default options 'bellow'.
		record (bool): Record the inference
	"""
	source = "video"			# [camera, video]
	cls = "1_on"				# [0_off, 1_on]

	device = 'cuda'				# [cpu, cuda]
	half = True					# [True, False]

	# resolution [SS, HD]
	resolution = resolution or (SS if (source == "camera") else HD)

	# Stream capture
	if source == "camera":
		stream = cv2.VideoCapture(idx)
		stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
		stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
	elif source == "video":
		# sample_path = f"samples/storage_videos/{cls}/"
		sample_path = f"samples/recorded/2023.11.03/original"
		samples = sorted(os.listdir(sample_path))
		stream = cv2.VideoCapture(f"{sample_path}/{samples[idx]}")

	# Recording
	if record:
		cur_date_time = datetime.now().strftime("%Y.%m.%d.%H-%M-%S")
		cur_date = ".".join(cur_date_time.split(".")[:3])

		path = "samples/recorded"
		path_org = f"{path}/{cur_date}/original"
		path_mod = f"{path}/{cur_date}/modified"
		Path(path_org).mkdir(parents=True, exist_ok=True)
		Path(path_mod).mkdir(parents=True, exist_ok=True)

		out_org = cv2.VideoWriter(f'{path_org}/{cur_date_time}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution)
		out_mod = cv2.VideoWriter(f'{path_mod}/{cur_date_time}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution)

	# Model loading
	deque_len = deque_len or 20
	model_path = f"weights/SEATBELT_resnet50_256_v4_jit_traced.pth"
	model = SeatbeltPTH(model_path=model_path, device=device, half=half, resize=256).load_model()

	occupant_state = {	
		'DR': True,
		'FP': True,
		'LP': True,
		'MP': True,
		'RP': True,
	}

	# roi_seatbelts = {	
	# 	'DR': (780, 220, 500, 500),		# (x, y, width, height)
	# 	'FP': (0, 400, 320, 320),
	# 	'LP': (325, 180, 150, 256),
	# 	'MP': (450, 180, 150, 256),
	# 	'RP': (600, 180, 150, 256),
	# }

	roi_seatbelts = {	
		'DR': (780, 220, 500, 500),		# (x, y, width, height)	updated
		'FP': (0,   400, 320, 320),
		'LP': (325, 190, 125, 185),
		'MP': (490, 200, 125, 185),
		'RP': (625, 190, 125, 185),
	}

	deque_save = {
		'DR': deque(maxlen=deque_len),
		'FP': deque(maxlen=deque_len),
		'LP': deque(maxlen=deque_len),
		'MP': deque(maxlen=deque_len),
		'RP': deque(maxlen=deque_len),
	}

	org_save = {
		'DR': (15, 50),
		'FP': (15, 50+35*1),
		'LP': (15, 50+35*2),
		'MP': (15, 50+35*3),
		'RP': (15, 50+35*4),
	}

	count = 0
	count_skip = 0
	count_warm_up = 5
	while (stream.isOpened):
		count += 1
		start = time()

		success, original = stream.read()

		if not success:
			break

		if count < count_skip:
			continue

		# Save original output
		frame = copy.copy(original)

		# Convert to RGB
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		temp_frame = copy.copy(frame)

		# Main process
		# ================================================================
		inference_speed_start = time()

		for i, (passenger, roi) in enumerate(roi_seatbelts.items()):
			if not occupant_state[passenger]: continue
			dev = 2 if resolution == SS else 1
			x, y, width, height = roi
			x, y, width, height = int(x/dev), int(y/dev), int(width/dev), int(height/dev)

			cropped = temp_frame[y:y+height, x:x+width]

			# Get prediction
			model_input = model.preprocess(cropped)
			model_output = model.predict(model_input)
			cls, confidence = model.postprocess(model_output, threshold=threshold)

			# Get average prediction
			deque_save[passenger].append(cls)
			most_frequent_cls = Counter(deque_save[passenger]).most_common(n=1)[0][0] if deque_save[passenger] else False

			# Draw results
			cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 255), 1, cv2.LINE_AA)
			color = (0, 255, 0) if most_frequent_cls == "1_on" else (255, 0, 0)
			cv2.putText(frame, f"{passenger}: {most_frequent_cls}", org_save[passenger], cv2.FONT_HERSHEY_SIMPLEX , 1, color, 2, cv2.LINE_AA)

		inference_speed_end = time()
		# ================================================================

		# Convert to BGR
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

		# Record a video
		if record:
			out_org.write(original)
			out_mod.write(frame)

		# Show the frame
		cv2.imshow("SeatBelts", frame)

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
	if record:
		out_org.release()
		out_mod.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
