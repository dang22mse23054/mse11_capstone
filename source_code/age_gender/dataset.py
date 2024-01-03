import sys, os
sys.path.append('../')

import torch
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from common.constants import Constants

UTK_FACE_PATH = '/kaggle/input/utkface-new/UTKFace'
UTK_FACE_PATH_DEMO = 'raw/UTKFace/'
AGE = Constants.Age()

TEST_RATIO = 0.1
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1

MIN_AGE, MAX_AGE = 1, 90
MIN_GENDER, MAX_GENDER = 0, 1
NUM_OF_AGE_GROUPS = len(AGE.Groups)

MODE = Constants.Mode()

def check_image(img_name):
	if '/' in img_name: img_name = img_name.split('/')[-1]
	meta_data = img_name.split('_')
	age, gender = int(meta_data[0]), int(meta_data[1])
	valid_gender = (gender >= MIN_GENDER and gender <= MAX_GENDER)
	valid_age = (age >= MIN_AGE and age <= MAX_AGE)
	
	return valid_age and valid_gender

class AgeGenderDataset(Dataset):
	def __init__(self, 
			  data_path=UTK_FACE_PATH,
			  mode : str = MODE.TRAIN,
			  transforms=None):

		self.mode = mode
		self.data_path = data_path
		self.transforms = transforms
		
		file_list = os.listdir(self.data_path)
		for img_name in file_list:
			if not check_image(img_name):
				file_list.remove(img_name)

		# splitting dataset
		testPart = int(len(file_list) * TEST_RATIO)
		trainPart = int(len(file_list) * TRAIN_RATIO)
		validationPart = int(len(file_list) * VALIDATION_RATIO)
		total = trainPart + validationPart + testPart
		trainPart += (len(file_list) - total)
		# check if the partition is correct
		if (trainPart + validationPart + testPart) == len(file_list): print("Correct Partition")

		# set file list depending on mode
		if self.mode == MODE.VALIDATE:
			self.file_list = file_list[trainPart:trainPart+validationPart]
		elif self.mode == MODE.TEST:
			self.file_list = file_list[trainPart+validationPart:]
		else:
			self.file_list = file_list[0:trainPart]

	def get_group(self, age):
		"""
		CHILDREN: 	(00~12) => 0
		TEENAGERS: 	(13~17) => 1
		ADULT: 		(18~44) => 2
		MIDDLE_AGED:(45~60) => 3
		ELDERLY: 	(61~12) => 4
		"""

		for idx, name in enumerate(AGE.Groups.keys()):
			group = AGE.Groups[name]
			if group['min'] <= age and age <= group['max']:
				return {'idx': idx, 'name': name, **group}

		else: raise ValueError(f"Age {age} does not fit inside any group.")

	def __getitem__(self, i):
		img_name, age, gender = self.extract_info(i)
		file_path = self.data_path + img_name
		image = cv2.imread(f'{file_path}', cv2.IMREAD_COLOR).astype(np.float32)

		# normalization.
		image /= 255.0
		target = {}

		# transformation
		if self.transforms: 
			# image = self.transforms(image)
			image = self.transforms(image=image)['image']
		
		image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
		
		# Solution 1
		age_tensor, gender_tensor = torch.tensor(age), torch.tensor(gender)
		
		# Solution 2
		# # create tensors for age and gender
		# age_tensor, gender_tensor = torch.zeros(NUM_OF_AGE_GROUPS), torch.zeros(2)
		# # set 
		# age_tensor[self.get_group(age)['idx']] = 1
		# gender_tensor[gender] = 1
		# Merge tensors.
		# target = torch.cat([age_tensor, gender_tensor], dim=0)

		target = (age_tensor, gender_tensor)

		return image, target

	def extract_info(self, i):
		img_name = self.file_list[i]
		
		# label processing
		if '/' in img_name: 
			img_name = img_name.split('/')[-1]
		meta_data = img_name.split('_')
		age, gender = int(meta_data[0]), int(meta_data[1])

		return img_name, age, gender

	def test_item(self, i):
		return self[i], self.file_list[i]

	def __len__(self):
		return len(self.file_list)
