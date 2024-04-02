import sys
sys.path.append('../')

from typing import Optional
from common.constants import Constants
import numpy as np
import common.utils as utils
from matplotlib import pyplot as plt
import albumentations as albu
from albumentations import Compose, Normalize, Resize, Rotate
from albumentations.pytorch import ToTensorV2

MODE = Constants.Mode()
AGE = Constants.Age()

# def display_info(dataset: AgeGenderDataset, name='', n_bins=AGE.ELDERLY['max']):
# 	ages, genders = [], []
# 	for i in range(len(dataset)):
# 		name, age, gender = dataset.extract_info(i)
# 		ages.append(age)
# 		genders.append(gender)

# 	plt.title(f'{name} Ages Distribution')
# 	plt.xlabel('Person Age')
# 	plt.ylabel('Number of Images')
# 	plt.hist(ages, n_bins)
# 	plt.show()

# 	plt.title(f'{name} Gender Distribution')
# 	plt.xlabel('Person Gender')
# 	plt.ylabel('Number of Images')
# 	genders = [genders.count(0), genders.count(1)]
# 	plt.bar([0, 1], genders)
# 	plt.show()

from dataset import AgeGenderDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

class AgeGenderDataLoader(LightningDataModule):
	def __init__(self,         
		batch_size: int = 48,
		workers: int = 5,
		img_size: int = 224,
	):
		super().__init__()
		self.batch_size = batch_size
		self.workers = workers
		# height, width
		self.img_size = (img_size, img_size)
		self.train_dataset = None
		self.val_dataset = None
		self.test_dataset = None
	
	def setup(self, stage: Optional[str] = None) -> None:

		augmentations = [
			albu.RandomResizedCrop(*self.img_size, scale=(0.6, 1)),
			albu.HorizontalFlip(),
			albu.RandomBrightnessContrast(),
			albu.OneOf([
				# albu.CLAHE(),
				albu.Blur(5),
				albu.RGBShift()  
			], p=1),
		]
		
		train_transforms = albu.Compose([
			*augmentations,
			albu.Normalize(),
			ToTensorV2()
		])

		valid_transforms = albu.Compose([
			albu.Resize(*(np.array(self.img_size) * 1.25).astype(int)),
			albu.CenterCrop(*self.img_size),
			albu.Normalize(),
			ToTensorV2()
		])

		test_transforms = albu.Compose([
			albu.Resize(*(np.array(self.img_size) * 1.25).astype(int)),
			albu.CenterCrop(*self.img_size),
			albu.Normalize(),
			ToTensorV2()
		])

		if stage == "fit" or stage is None:
			self.train_dataset = AgeGenderDataset(mode=MODE.TRAIN, transforms=train_transforms)
			self.val_dataset = AgeGenderDataset(mode=MODE.VALIDATE, transforms=valid_transforms)
	   
		if stage == "test":
			self.test_dataset = AgeGenderDataset(mode=MODE.TEST, transforms=test_transforms)

	def train_dataloader(self):
		train_loader = DataLoader(
			dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True,
			num_workers=self.workers, #collate_fn=utils.collate_fn,
			persistent_workers=True, 
		)
		return train_loader

	def val_dataloader(self):
		val_loader = DataLoader( 
			dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False,
			num_workers=self.workers, #collate_fn=utils.collate_fn,
			persistent_workers=True, 
		)
		return val_loader
	
	def test_dataloader(self):
		test_loader = DataLoader( 
			dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False,
			num_workers=self.workers, #collate_fn=utils.collate_fn,
			persistent_workers=True, 
		)
		return test_loader

