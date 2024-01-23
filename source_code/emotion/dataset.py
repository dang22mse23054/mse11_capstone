import sys, os, random
sys.path.append('../')

import torch
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from common.constants import Constants
import matplotlib.pyplot as plt
import albumentations as albu


EMOTION_PATH = '/kaggle/input/fer2013'
# FOR DEBUG
# EMOTION_PATH = 'raw'

MODE = Constants.Mode()

PATHS = {
	MODE.TRAIN: f'{EMOTION_PATH}/train',
	MODE.TEST:  f'{EMOTION_PATH}/test'
}

EMOTION = Constants.Emotion()
EMOTION_GROUPS = EMOTION.Groups

VALIDATION_RATIO = 0.2

def check_images(file_list, title = 'Graph'):
	cols = 4
	rows = int(len(file_list) / cols) + 1
	fig, axes = plt.subplots(rows, cols, figsize=(10, 7))
	# Flatten axes để dễ quản lý
	axes = axes.flatten()

	with torch.no_grad():
		for idx, file_path in enumerate(file_list):

			# Read the list of image files
			input_image = Image.open(file_path).convert('RGB')
			emotion_name = file_path.split('/')[-2]
			emotion_id = EMOTION_GROUPS.index(emotion_name)
			axes[idx].set_title(f'{emotion_name} ({emotion_id})')
			axes[idx].imshow(input_image, cmap='gray')
			axes[idx].axis('off')
		
	plt.tight_layout()
	plt.title(title, fontdict={'size': 16, 'color': 'red'}) 
	plt.show()

# Dataset FER2013
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

		check_images(self.file_list[:8], f'{self.mode} set ({len(self.file_list)} items)')

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

			return train_set if self.mode == MODE.TRAIN else val_set
		
	def __getitem__(self, idx):
		file_path = self.file_list[idx]
		emotion_name = file_path.split('/')[-2]
		
		emotion_id = EMOTION_GROUPS.index(emotion_name)
		# phải chuyển thành tensor để có thể tính loss function
		emotion_id_tensor = torch.tensor(emotion_id)

		image = Image.open(file_path).convert('RGB')

		# normalization.
		image = np.array(image, dtype=np.float32) / 255.0
		
		# transformation
		if self.transforms: 
			# image = self.transforms(image)
			image = self.transforms(image=image)['image']
		
		# đã dùng TensorV2 thì ko cần dòng này
		# image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)

		return image, emotion_id_tensor
	
	def __len__(self):
		return len(self.file_list)
