import sys
sys.path.append('../')

import numpy as np
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data.dataloader import DataLoader
import albumentations as albu
from albumentations import Compose, RandomCrop, BboxParams
from albumentations.pytorch import ToTensorV2
from dataset import FaceDetectionDataset
from common.constants import Constants
import common.utils as utils

MODE = Constants.Mode()

class FaceDataLoader(pl.LightningDataModule):
	def __init__(self,         
		batch_size: int = 10,
		workers: int = 5,
		img_size: int = 224,
		crop_size: int = 170,
	):
		super().__init__()
		self.batch_size = batch_size
		self.workers = workers
		# height, width
		self.img_size = (img_size, img_size)
		self.crop_size = (img_size, img_size)
		self.train_dataset = None
		self.val_dataset = None
		self.test_dataset = None
	
	def setup(self, stage: Optional[str] = None) -> None:
		if stage == "fit" or stage is None:
			# transforms = albu.Compose(
			# 	[
			# 		albu.RandomCrop(*self.img_size, p=1.0),
			# 	], 
			# 	bbox_params=albu.BboxParams(format='pascal_voc', min_visibility=0.85, label_fields=None)
			# )

			augmentations = [
				albu.RandomCrop(*self.crop_size, p=1.0),
				# albu.Resize(*self.img_size),
				# albu.HorizontalFlip(),
				# albu.RandomBrightnessContrast(),
				# albu.OneOf([
				# 	# albu.CLAHE(),
				# 	albu.Blur(5),
				# 	albu.RGBShift()  
				# ], p=1),
			]
			
			train_transforms = albu.Compose([
				*augmentations,
				albu.Normalize(),
				ToTensorV2()
			], bbox_params=albu.BboxParams(format='pascal_voc', min_visibility=0.85, label_fields=None))

			valid_transforms = albu.Compose([
				albu.Resize(*(np.array(self.img_size) * 1.25).astype(int)),
				albu.CenterCrop(*self.img_size),
				albu.Normalize(),
				ToTensorV2()
			], bbox_params=albu.BboxParams(format='pascal_voc', min_visibility=0.85, label_fields=None))

			# test_transforms = albu.Compose([
			# 	albu.Resize(*(np.array(self.img_size) * 1.25).astype(int)),
			# 	albu.CenterCrop(*self.img_size),
			# 	albu.Normalize(),
			# 	ToTensorV2()
			# ], bbox_params=albu.BboxParams(format='pascal_voc', min_visibility=0.85, label_fields=None))

			self.train_dataset = FaceDetectionDataset(mode=MODE.TRAIN, transforms=train_transforms)
			self.val_dataset = FaceDetectionDataset(mode=MODE.VALIDATE, transforms=valid_transforms)
	   
		# if stage == "test" or stage is None:
		# 	self.test_dataset = FaceDetectionDataset(mode=MODE.TEST, image_dir=self.val_data)

	def train_dataloader(self):
		train_loader = DataLoader(
			dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True,
			num_workers=self.workers, collate_fn=utils.collate_fn,
			persistent_workers=True, 
		)
		return train_loader

	def val_dataloader(self):
		val_loader = DataLoader( 
			dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False,
			num_workers=self.workers, collate_fn=utils.collate_fn,
			persistent_workers=True, 
		)
		return val_loader
	
	def test_dataloader(self):
		return self.val_dataloader()

