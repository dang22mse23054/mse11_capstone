import os
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed

import re, subprocess

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from faster_rcnn import FasterRCNNResNet50FPN as Model
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.optim.lr_scheduler as lr_scheduler



class FaceDetectionModel(LightningModule):
	def __init__(self,
				lr: float = 1e-3,
				momentum: float = 0.9,
				weight_decay: float = 1e-4,
				**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		self.id2label = {0: 'Background', 1: 'Face'}
		# metrics
		self.model = Model(num_classes=2)
	
	def forward(self, x):
		return self.model(x)
	
	def training_step(self, batch, batch_idx):
		if len(batch) == 0 : return torch.tensor(0.)
		images, targets = batch
		targets = [{k: v for k, v in t.items()} for t in targets]
		loss_dict = self.model(images, targets)

		self.log(
			"train_loss_classifier",
			loss_dict["loss_classifier"],
			on_step=True,
			on_epoch=True,
			prog_bar=False,
			logger=True,
		)
		self.log(
			"train_loss_box_reg",
			loss_dict["loss_box_reg"],
			on_step=True,
			on_epoch=True,
			prog_bar=False,
			logger=True,
		)

		self.log(
			"train_loss_objectness",
			loss_dict["loss_objectness"],
			on_step=True,
			on_epoch=True,
			prog_bar=False,
			logger=True,
		)

		self.log(
			"train_loss_rpn_box_reg",
			loss_dict["loss_rpn_box_reg"],
			on_step=True,
			on_epoch=True,
			prog_bar=False,
			logger=True,
		)

		# total loss
		losses = sum(loss for loss in loss_dict.values())
		self.log('train_loss', losses, prog_bar=True, on_step=True, on_epoch=True)

		return losses

	def eval_step(self, batch, batch_idx, prefix: str):
		import random
		# if random.random() < 0.1:
		if len(batch) == 0: return
		images, targets = batch
		preds = self.model(images)
		selected = random.sample(range(len(images)), len(images) // 5)
		self.mAP.update([preds[i] for i in selected], [targets[i] for i in selected])
    
	def on_validation_epoch_start(self):
		self.mAP = MeanAveragePrecision(box_format="xyxy", class_metrics=False)

	def on_validation_epoch_end(self) -> None:
		self.mAPs = {"val_" + k: v for k, v in self.mAP.compute().items()}
		self.print(self.mAPs)
		self.log_dict(self.mAPs, sync_dist=True)

		self.mAP.reset()

	def validation_step(self, batch, batch_idx):
		return self.eval_step(batch, batch_idx, "val")
	
	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
		# We will reduce the learning rate by 0.1 after 20 epochs
		scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 20))
		return [optimizer], [scheduler]