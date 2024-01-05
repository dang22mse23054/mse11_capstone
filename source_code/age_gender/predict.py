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

NUM_OF_AGE_GROUPS = len(AGE.Groups)

TEST_RATIO = 0.1

def init_model():
	checkpoint = torch.load("checkpoint/epoch=6-val_loss=0.1157.ckpt", map_location=torch.device('cpu'))
	model = AgeGenderDetectionModel(
		lr=1e-4,
		momentum=0.9,
		weight_decay=1e-4,
	)
	model.load_state_dict(checkpoint['state_dict'])
	
	# Set Model to Evaluation Mode
	model.eval()
	return model

def predict_all(file_list, model):
	predictions = []
	transforms = albu.Compose([
		# albu.ToTensor(),
		albu.Normalize(mean=(0.5,), std=(0.5,)),
		albu.Resize(200, 200),
		albu.Rotate(limit=10),
		ToTensorV2()
	])
	with torch.no_grad():
		for idx, img_name in enumerate(file_list):

			# Read the list of image files
			file_path = UTK_FACE_PATH + img_name
			# input_image = cv2.imread(f'{file_path}', cv2.IMREAD_COLOR).astype(np.float32)
			input_image = Image.open(file_path).convert('RGB')

			# Preprocess the image using torchvision.transforms
			# preprocess = transforms.Compose([
			# 	# transforms.Resize((224, 224)),
			# 	transforms.ToTensor(),
			# 	# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			# ])
			# input_tensor = preprocess(input_image)

			input_tensor = transforms(image=np.array(input_image))['image']
			input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
			
			prediction = model(input_batch)

			print(prediction)

			pred_gender = int(torch.argmax(prediction[0]))
			pred_age = int(torch.argmax(prediction[1]))

			target = (img_name, pred_gender, pred_age)

			predictions.append(target)

			# input_image.show()
			print(f"Gender = {pred_gender} => Age = {pred_age}")

			plt.subplot(len(file_list), len(file_list), idx+1)
			plt.title(f'{list(AGE.Groups.keys())[pred_age]} {"Male" if pred_gender == 0 else "Female"}')
			plt.imshow(input_image, cmap='gray')
			plt.axis('off')
		plt.show()

	
	return predictions

if __name__ == "__main__":
	file_list = os.listdir(UTK_FACE_PATH)
	model = init_model()
	
	# splitting dataset
	testPart = int(len(file_list) * TEST_RATIO)
	# file_list = file_list[len(file_list) - testPart:]

	file_list = random.choices(file_list, k=2)
	
	# Make predictions for all files in the list
	all_predictions = predict_all(file_list, model)

	print(all_predictions)
	
	# Get the predictions images
	# for file_path, prediction in zip(file_list, all_predictions):
	# 	file_path = f'{PATHS[MODE.TESTDEMO]["img_dir"]}/{file_path}'
	# 	input_image = Image.open(file_path).convert('RGB')
	# 	# show_bbox(input_image, prediction)
	# 	face_images = extract_faces(input_image, prediction)
	# 	# get the face features
	# 	features = get_features(face_images, model)
		
	# 	print('---------- FACE features ----------')
	# 	print(features)
	
	