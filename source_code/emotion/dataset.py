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

EMOTION_PATH = 'kaggle/input/fer2013'
# FOR DEBUG
# EMOTION_PATH = 'raw/'

MODE = Constants.Mode()

PATHS = {
	MODE.TRAIN: f'{EMOTION_PATH}/train',
	MODE.TEST:  f'{EMOTION_PATH}/test'
}

EMOTION = Constants.Emotion()
EMOTION_GROUPS = EMOTION.Groups

VALIDATION_RATIO = 0.2

class EmotionDataset(Dataset):
	def __init__(self, 
			  mode : str = MODE.TRAIN,
			  transforms=None):

		self.mode = mode
		self.data_path = PATHS[MODE.TEST if mode == MODE.TEST else MODE.TRAIN] 
		self.transforms = transforms

		# splitting dataset
		self.file_list = self.init_dataset()
		random.Random(4).shuffle(self.file_list)


	def __getitem__(self, idx):
		file_path = self.file_list[idx]
		emotion_name = file_path.split('/')[-2]
		emotion_id = EMOTION_GROUPS.index(emotion_name)

		image = Image.open(file_path).convert('RGB')

		# normalization.
		image = np.array(image, dtype=np.float32) / 255.0
		
		# transformation
		if self.transforms: 
			# image = self.transforms(image)
			image = self.transforms(image=image)['image']
		
		# đã dùng TensorV2 thì ko cần dòng này
		# image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)

		return image, emotion_id
	
	def init_dataset(self):
		train_set = []
		val_set = []
		test_set = []
		folder_path = os.walk(self.data_path)

		if self.mode == MODE.TEST:
			for dirpath, dirnames, filenames in folder_path:
				total = int(len(filenames))
				if (total > 0):
					test_set.extend([os.path.join(dirpath, filename) for filename in filenames])

			return test_set
		else:
			for dirpath, dirnames, filenames in folder_path:
				total = int(len(filenames))

				if (total > 0):
					validation_part = int(len(filenames) * VALIDATION_RATIO)
					train_part = total - validation_part
					train_set.extend([os.path.join(dirpath, filename) for filename in filenames[:train_part]])
					val_set.extend([os.path.join(dirpath, filename) for filename in filenames[train_part:]])
					val_set.extend(filenames[train_part:])

			return train_set if self.mode == MODE.TRAIN else val_set
		
	def __len__(self):
		return len(self.file_list)
