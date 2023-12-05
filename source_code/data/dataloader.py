from typing import Optional
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from dataset import ImageDetectionDataset
import pytorch_lightning as pl
from albumentations import Compose, RandomCrop, BboxParams
from constants import Constants
import utils as utils

MODE = Constants.Mode()

class FaceDataLoader(pl.LightningDataModule):
	def __init__(self,         
		batch_size: int = 10,
		workers: int = 5,
		img_size: int = 256,
		):
		super().__init__()
		self.batch_size = batch_size
		self.workers = workers
		self.img_size = img_size
		self.train_dataset = None
		self.val_dataset = None
		self.test_dataset = None
	
	def setup(self, stage: Optional[str] = None) -> None:
		if stage == "fit" or stage is None:
			transforms = Compose([
							RandomCrop(self.img_size, self.img_size, p=1.0),
							], bbox_params=BboxParams(format='pascal_voc', min_visibility=0.85, label_fields=None))

			self.train_dataset = ImageDetectionDataset(mode=MODE.TRAIN, transforms=transforms)
			self.val_dataset = ImageDetectionDataset(mode=MODE.VALIDATE, transforms=transforms)
	   
		# if stage == "test" or stage is None:
		# 	self.test_dataset = ImageDetectionDataset(mode=MODE.TEST, image_dir=self.val_data)

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

