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
EMOTION = Constants.Emotion()

from dataset import EmotionDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

class EmotionDataLoader(LightningDataModule):
	def __init__(self,         
		batch_size: int = 64,
		workers: int = 4,
		img_size: int = 48,
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
			self.train_dataset = EmotionDataset(mode=MODE.TRAIN, transforms=train_transforms)
			self.val_dataset = EmotionDataset(mode=MODE.VALIDATE, transforms=valid_transforms)
	   
		if stage == "test":
			self.test_dataset = EmotionDataset(mode=MODE.TEST, transforms=test_transforms)

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

