import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import random
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.optim.lr_scheduler as lr_scheduler
from models.age_gender_resnet50 import AgeGenderResNet50

def accuracy(pred: torch.Tensor, gt: torch.Tensor):
    """
    accuracy metric
    
    expects pred shape bs x n_c, gt shape bs x 1
    """
    return (pred.max(1)[1] == gt).float().mean()

# Ref: 
# https://github.com/Sklyvan/Age-Gender-Prediction/blob/main/Age%20%26%20Gender%20Prediction%20Model%20Creation.ipynb
# https://github.com/thepowerfuldeez/age_gender_classifier/blob/master/training.ipynb

class AgeGenderDetectionModel(LightningModule):
	def __init__(self,
				encoder_channels: int = 2048,
				output_channels: int = 512,
				age_classes: int = 5,
				gender_classes: int = 2,
				lr: float = 1e-3,
				momentum: float = 0.9,
				weight_decay: float = 1e-4,
				**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		
		self.model = AgeGenderResNet50(
			encoder_channels,
			output_channels,
			age_classes,
			gender_classes,
		)
		self.loss_functions = {
			# 'Age': nn.MSELoss(),
			# 'Gender': nn.BCEWithLogitsLoss()
			'Age': nn.BCEWithLogitsLoss(),
			'Gender': nn.CrossEntropyLoss()
		}
	
	def forward(self, x):
		return self.model(x)

	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
		# We will reduce the learning rate by 0.1 after 20 epochs
		scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 20))
		# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)

		return [optimizer], [scheduler]
	
	# TODO: Q&A
	def on_validation_epoch_start(self):
		self.gender_acc_list = []
		self.age_acc_list = []
	
	# TODO: Q&A
	def on_validation_epoch_end(self) -> None:
		gender_acc = np.mean(self.gender_acc_list)
		print(self.gender_acc_list)
		print(f"gender_acc = {gender_acc}")
		age_acc = np.mean(self.age_acc_list)
		print(f"age_acc = {age_acc}")
		print(self.age_acc_list)
		# print(f"val epoch {epoch}, gender acc {gender_acc:.2%}, age acc {age_acc:.2%}")
		loss = (gender_acc + age_acc) / 2
		print(f"Loss = {loss}")
		self.log('val_loss', loss, prog_bar=True, on_epoch=True)

		self.gender_acc_list.clear()
		self.age_acc_list.clear()

	# def on_test_epoch_start(self) -> None:
	# 	gender_acc = np.mean(self.gender_acc_list)
	# 	age_acc = np.mean(self.age_acc_list)
	# 	# print(f"val epoch {epoch}, gender acc {gender_acc:.2%}, age acc {age_acc:.2%}")
	# 	loss = (gender_acc + age_acc) / 2
	# 	self.log('val_loss', loss, prog_bar=True, on_epoch=True)

	# 	self.gender_acc_list.clear()
	# 	self.age_acc_list.clear()
	
	# def on_test_epoch_end(self) -> None:
	# 	gender_acc = np.mean(self.gender_acc_list)
	# 	age_acc = np.mean(self.age_acc_list)
	# 	# print(f"val epoch {epoch}, gender acc {gender_acc:.2%}, age acc {age_acc:.2%}")
	# 	loss = (gender_acc + age_acc) / 2
	# 	self.log('val_loss', loss, prog_bar=True, on_epoch=True)

	# 	self.gender_acc_list.clear()
	# 	self.age_acc_list.clear()

	def training_step(self, batch, batch_idx):
		# if len(batch) == 0 : return torch.tensor(0.)
		if len(batch) == 0 : return torch.tensor(0.0, requires_grad=True)

		image, (gender_gt, age_gt) = batch
		gender_logits, age_logits = self(image)
		# print(f"Gender = {gender_gt} => Logits = {gender_logits}")
		# print(f"Age = {age_gt} => Logits = {age_logits}")
		
		# BCE expects one-hot vector
		age_gt_onehot = torch.zeros(*age_logits.size(), device=age_logits.device)
		age_gt_onehot = age_gt_onehot.scatter_(1, age_gt.unsqueeze(-1).long(), 1)
		loss_age = self.loss_functions['Age'](age_logits, age_gt_onehot)  # bce
		
		gender_gt = gender_gt.long()
		loss_gender = self.loss_functions['Gender'](gender_logits, gender_gt)  # softmax+ce
		
		losses = (loss_gender + loss_age) / 2

		self.log('train_loss', losses, prog_bar=True, on_step=True, on_epoch=True)

		return losses


	def eval_step(self, batch, batch_idx, prefix: str):
		# if random.random() < 0.1:
		if len(batch) == 0: return

		image, (gender_gt, age_gt) = batch

		gender_logits, age_logits = self(image)
		self.gender_acc_list.append(accuracy(gender_logits, gender_gt).item())
		self.age_acc_list.append(accuracy(age_logits, age_gt).item())
		

	def validation_step(self, batch, batch_idx):
		return self.eval_step(batch, batch_idx, "val")
	
	def test_step(self, batch, batch_idx):
		image, (gender_gt, age_gt) = batch
		print('---------test_step---------')
		print(gender_gt)
		print(age_gt)

		# real_age = int(torch.argmax(y[0][:Classes]))
        # read_gender = int(torch.argmax(y[0][Classes:]))

		# # implement your own
		# out = self(x)
		# loss = self.loss(out, y)

		# # log 6 example images
		# # or generated text... or whatever
		# sample_imgs = x[:6]
		# grid = torchvision.utils.make_grid(sample_imgs)
		# self.logger.experiment.add_image('example_images', grid, 0)

		# # calculate acc
		# labels_hat = torch.argmax(out, dim=1)
		# test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

		# # log the outputs!
		# self.log_dict({'test_loss': loss, 'test_acc': test_acc})