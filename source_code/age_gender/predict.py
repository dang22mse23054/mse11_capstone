import sys
import argparse, random
import os
sys.path.append('../')

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import AgeGenderDetectionModel
from torchvision import transforms
from PIL import ImageDraw
from common.constants import Constants
from dataset import UTK_FACE_PATH
from albumentations import Compose, RandomCrop, BboxParams
import albumentations as albu
from albumentations.pytorch import ToTensorV2

MODE = Constants.Mode()
AGE = Constants.Age()
AGES = list(Constants.Age().Groups.keys())
NUM_OF_AGE_GROUPS = len(AGES)


TEST_IMG_PATH = 'raw/test/' #UTK_FACE_PATH

TEST_RATIO = 0.1

def init_model():
	checkpoint = torch.load("checkpoint/epoch=24-val_acc=0.8411-val_age_acc=0.8013-val_gender_acc=0.8857.ckpt", map_location=torch.device('cpu'))
	model = AgeGenderDetectionModel()
	model.load_state_dict(checkpoint['state_dict'])
	
	# Set Model to Evaluation Mode
	model.eval()
	return model

def predict_all(file_list, model):
	predictions = []

	img_size = (224, 224)
	transforms = albu.Compose([
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
		for idx, img_name in enumerate(file_list):

			# Read the list of image files
			file_path = TEST_IMG_PATH + img_name
			input_image = Image.open(file_path).convert('RGB')

			# Preprocess the image using torchvision.transforms
			input_tensor = transforms(image=np.array(input_image)/255)['image']
			input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
			
			prediction = model(input_batch)

			print(prediction)

			pred_gender = int(torch.argmax(prediction[0]))
			pred_age = int(torch.argmax(prediction[1]))

			target = (img_name, pred_gender, pred_age)

			predictions.append(target)

			# input_image.show()
			print(f"Gender = {pred_gender} => Age = {pred_age}")

			axes[idx].set_title(f'{AGES[pred_age]} {"Male" if pred_gender == 0 else "Female"}')
			axes[idx].imshow(input_image, cmap='gray')
			axes[idx].axis('off')
		
	plt.tight_layout()
	plt.show()

	
	return predictions

if __name__ == "__main__":
	file_list = os.listdir(TEST_IMG_PATH)
	model = init_model()
	
	# splitting dataset
	testPart = int(len(file_list) * TEST_RATIO)
	# file_list = file_list[len(file_list) - testPart:]

	file_list = random.sample(file_list, k=8)
	# print(file_list)

	# Make predictions for all files in the list
	all_predictions = predict_all(file_list, model)
	# print(all_predictions)
	
	