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
		self.map = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
		
		self.model = Model(num_classes=2)
		print(f'Support MPS = {torch.backends.mps.is_available()}')
		print(f'PyTorch built with MPS activated: {torch.backends.mps.is_built()}')


		cpu_info = subprocess.run(["system_profiler","SPHardwareDataType"], stdout=subprocess.PIPE).stdout.decode("utf-8")
		gpu_info = subprocess.run(["system_profiler","SPDisplaysDataType"], stdout=subprocess.PIPE).stdout.decode("utf-8") 

		cpu = re.search(r'Chip:\s+(.+)', cpu_info).group(1)
		cpu_cores = re.search(r'Number of Cores:\s+(\d+)', cpu_info).group(1)
		memory = re.search(r'Memory:\s+(\d+)\s+GB', cpu_info).group(1)

		print(cpu, cpu_cores, memory)

		gpu = re.search(r'Chipset Model:\s+(.+)', gpu_info).group(1)
		gpu_cores = re.search(r'Total Number of Cores:\s+(\d+)', gpu_info).group(1)

		print(gpu, gpu_cores)

	
	def forward(self, x):
		print('-----------forward------')

		return self.model(x)

	# def training_step(self, batch, batch_idx):
	# 	inputs, target = batch
	# 	output = self.model(inputs, target)
	# 	loss = torch.nn.functional.nll_loss(output, target.view(-1))
	# 	return loss
	
	def training_step(self, batch, batch_idx):
		print(f'-----------training_step------ {batch_idx}')

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
		print('-----------training_step  _0------')

		losses = sum(loss for loss in loss_dict.values())
		print('-----------training_step  _1------')


		self.log('train_loss', losses, prog_bar=True, on_step=True, on_epoch=True)

		# debug: skip calculating loss
		losses = torch.tensor(0.)
		losses.requires_grad = True
		print(f'-----------training_step  _loss ------ {losses}')

		return losses

	def eval_step(self, batch, batch_idx, prefix: str):
		print('-----------eval_step------')

		import random
		if random.random() < 0.1:
			images, targets = batch
			preds = self.model(images)
			
			selected = random.sample(range(len(images)), len(images) // 5)
			

			print([targets[i] for i in selected])


			print('---------------')

			self.map.update([preds[i] for i in selected], [targets[i] for i in selected])

	def validation_step(self, batch, batch_idx):
		return self.eval_step(batch, batch_idx, "val")
	
	def configure_optimizers(self):
		print('-----------configure_optimizers------')

		optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
		# We will reduce the learning rate by 0.1 after 20 epochs
		scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 20))
		return [optimizer], [scheduler]