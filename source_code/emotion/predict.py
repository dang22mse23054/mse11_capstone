import sys
import argparse, random
import os
import os.path
sys.path.append('../')

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import EmotionDetectionModel
from torchvision import transforms
from PIL import ImageDraw
from common.constants import Constants
from albumentations import Compose, RandomCrop, BboxParams
import albumentations as albu
from albumentations.pytorch import ToTensorV2

MODE = Constants.Mode()
EMOTION = Constants.Emotion()

TEST_RATIO = 0.1

def init_model():
	checkpoint = torch.load("checkpoint/epoch=13-val_acc=0.7565.ckpt", map_location=torch.device('cpu'))
	model = EmotionDetectionModel()
	model.load_state_dict(checkpoint['state_dict'])
	
	# Set Model to Evaluation Mode
	model.eval()
	return model

def predict_all(file_list, model):
	predictions = []

	img_size = (48, 48)
	transforms = albu.Compose([
		# albu.Resize(*img_size),
		albu.Resize(*(np.array(img_size) * 1.25).astype(int)),
		albu.CenterCrop(*img_size),
		albu.Normalize(),
		ToTensorV2()
	])

	cols = 4
	rows = int(len(file_list) / cols) + 1
	fig, axes = plt.subplots(rows, cols, figsize=(10, 7))
	# Flatten axes để dễ quản lý
	axes = axes.flatten()

	with torch.no_grad():
		for idx, file_path in enumerate(file_list):

			# Read the list of image files
			parts = file_path.split("/")
			img_name = parts[-1]
			true_label = parts[-2]
			input_image = Image.open(file_path).convert('RGB')

			# Preprocess the image using torchvision.transforms
			input_tensor = transforms(image=np.array(input_image)/255)['image']
			input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
			
			prediction = model(input_batch)

			print(prediction)

			pred_emotion = int(torch.argmax(prediction[0]))

			target = (img_name, pred_emotion)

			predictions.append(target)

			# input_image.show()
			print(f"Emotion = {pred_emotion}")

			axes[idx].set_title(f'{EMOTION.Groups[pred_emotion]} (#{true_label})')
			axes[idx].imshow(input_image, cmap='gray')
			axes[idx].axis('off')
		
	plt.tight_layout()
	plt.show()

	
	return predictions

if __name__ == "__main__":

	FODLER_PATH = './raw/fer2013-org/test'
	subfolder_list = os.listdir(FODLER_PATH)
	file_list = []
	for folder_name in subfolder_list:
		folder_path = f'{FODLER_PATH}/{folder_name}'

		if folder_name.startswith('.'):
			continue
		
		files = random.sample(os.listdir(folder_path), k=1)
		file_list.append(f'{folder_path}/{files[0]}')

	model = init_model()
	
	# splitting dataset
	# testPart = int(len(file_list) * TEST_RATIO)
	# file_list = file_list[len(file_list) - testPart:]

	# file_list = random.sample(file_list, k=6)
	# print(file_list)

	# Make predictions for all files in the list
	all_predictions = predict_all(file_list, model)
	# print(all_predictions)
	
	