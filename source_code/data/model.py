import os
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed


import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from faster_rcnn import FasterRCNNResNet50FPN as Model
# from torchmetrics.detection.map import MeanAveragePrecision
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
		# self.map = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
		self.model = Model(num_classes=2)
	
	def forward(self, x):
		return self.model(x)

	# def training_step(self, batch, batch_idx):
	# 	inputs, target = batch
	# 	output = self.model(inputs, target)
	# 	loss = torch.nn.functional.nll_loss(output, target.view(-1))
	# 	return loss
	
	def training_step(self, batch, batch_idx):
		print('-----------')
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

	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
		scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 20))
		return [optimizer], [scheduler]