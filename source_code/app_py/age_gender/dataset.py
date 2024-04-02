import sys, os, random
sys.path.append('../')

import torch
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from common.constants import Constants

UTK_FACE_PATH = '/kaggle/input/utkface-new/UTKFace/'
# FOR DEBUG
UTK_FACE_PATH = 'raw/UTKFace/'

AGE = Constants.Age()

TEST_RATIO = 0.1
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1

MIN_AGE, MAX_AGE = 1, 120
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
		random.Random(4).shuffle(file_list)
		for img_name in file_list:
			if not check_image(img_name):
				file_list.remove(img_name)

		# splitting dataset
		testPart = int(len(file_list) * TEST_RATIO)
		trainPart = int(len(file_list) * TRAIN_RATIO)
		validationPart = int(len(file_list) * VALIDATION_RATIO)
		total = trainPart + validationPart + testPart
		trainPart += (len(file_list) - total)

		print(f'(Mode {self.mode}) Total: {len(file_list)}, Train: {trainPart}, Validation: {validationPart}, Test: {testPart}')

		# check if the partition is correct
		if (trainPart + validationPart + testPart) == len(file_list): print(f"(Mode {self.mode}) Correct Partition")

		# set file list depending on mode
		if self.mode == MODE.TESTDEMO:
			self.file_list = file_list[trainPart:trainPart+validationPart]
		elif self.mode == MODE.VALIDATE:
			self.file_list = file_list[trainPart+validationPart:]
			# self.file_list = random.sample(file_list, k=3)

		else:
			self.file_list = file_list[0:trainPart]

		df = pd.DataFrame(self.file_list, columns=['List'])
		print(df)

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
			if group['from'] <= age and age <= group['to']:
				return {'idx': idx, 'name': name, **group}

		else: raise ValueError(f"Age {age} does not fit inside any group.")

	def __getitem__(self, i):
		img_name, gender, age = self.extract_info(i)
		file_path = self.data_path + img_name
		image = Image.open(file_path).convert('RGB')

		# normalization.
		image = np.array(image, dtype=np.float32) / 255.0
		
		target = {}

		# transformation
		if self.transforms: 
			# image = self.transforms(image)
			image = self.transforms(image=image)['image']
		
		# đã dùng TensorV2 thì ko cần dòng này
		# image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
		
		# Solution 1
		# convert age to group
		age = self.get_group(age)['idx']
		age_tensor, gender_tensor = torch.tensor(age), torch.tensor(gender)
		
		# Solution 2
		# # create tensors for age and gender
		# age_tensor, gender_tensor = torch.zeros(NUM_OF_AGE_GROUPS), torch.zeros(2)
		# # set 
		# age_tensor[self.get_group(age)['idx']] = 1
		# gender_tensor[gender] = 1
		
		target = (gender_tensor, age_tensor)
		# Merge tensors.
		# target = torch.cat([age_tensor, gender_tensor], dim=0)

		# print(f"REAL Gender = {gender} => Logits = {age}")

		return image, target

	def extract_info(self, i):
		img_name = self.file_list[i]
		
		# label processing
		if '/' in img_name: 
			img_name = img_name.split('/')[-1]
		meta_data = img_name.split('_')
		age, gender = int(meta_data[0]), int(meta_data[1])

		return img_name, gender, age

	def __len__(self):
		return len(self.file_list)
