import os
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import random

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from models.faster_rcnn import FasterRCNNResNet50FPN
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.optim.lr_scheduler as lr_scheduler
from common.constants import Constants
FACE_GROUPS = Constants.FACE_GROUPS

class FaceDetectionModel(LightningModule):
	def __init__(self,
				lr: float = 1e-3,
				momentum: float = 0.9,
				weight_decay: float = 1e-4,
				**kwargs
	):
		super().__init__()
		self.lr = lr 
		self.momentum = momentum
		self.weight_decay = weight_decay 

		self.save_hyperparameters()

		self.model = FasterRCNNResNet50FPN(num_classes=len(FACE_GROUPS))

	def forward(self, x): 
		# Call the parent class forward method
		output = self.model(x)
		# Add feature map to the output
		if (len(output) > 0):
			output[0]['feature_map'] = self.model.feature_map_callback.feature_map
			
		return output

	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
		# We will reduce the learning rate by 0.1 after 20 epochs
		scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 20))
		return [optimizer], [scheduler]
	
	# TODO: Q&A
	def on_validation_epoch_start(self):
		self.mAP = MeanAveragePrecision(box_format="xyxy", class_metrics=False)
	
	# TODO: Q&A
	def on_validation_epoch_end(self) -> None:
		print('on_validation_epoch_end --- END ---')
		self.mAPs = {"val_" + k: v for k, v in self.mAP.compute().items()}
		self.log_dict(self.mAPs, sync_dist=True)
		self.mAP.reset()
	
	def training_step(self, batch, batch_idx):
		# if len(batch) == 0 : return torch.tensor(0.)
		if len(batch) == 0 : return torch.tensor(0.0, requires_grad=True)

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

		# debug: skip calculating loss
		# losses = torch.tensor(0.)
		# losses.requires_grad = True

		return losses

	def eval_step(self, batch, batch_idx, prefix: str):
		# if random.random() < 0.1:
		if len(batch) == 0:
			print('eval_step no batch item ')
			self.mAP.update([], [])
			return

		print('eval_step --- UPDATE ---')
		images, targets = batch
		preds = self.model(images)
		selected = random.sample(range(len(images)), len(images) // 5)
		print(f'eval_step --- len(selected) = {len(selected)} ---')
		self.mAP.update([preds[i] for i in selected], [targets[i] for i in selected])

	def validation_step(self, batch, batch_idx):
		return self.eval_step(batch, batch_idx, "val")