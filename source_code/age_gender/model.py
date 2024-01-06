import os, random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import random
from pytorch_lightning.core import LightningModule
import torch.optim.lr_scheduler as lr_scheduler
from models.age_gender_resnet50 import AgeGenderResNet50
import matplotlib.pyplot as plt
from common.constants import Constants

def accuracy(pred: torch.Tensor, gt: torch.Tensor):
	"""
	accuracy metric

	expects pred shape bs x n_c, gt shape bs x 1

	Ví dụ:
		input là pred: có kích thước bs x n_c
		pred = tensor([[0.1, 0.9],
					   [0.8, 0.2],
					   [0.3, 0.7]])
		thì pred.max(1) sẽ là .object có format 
		(
			values=tensor([0.7000, 0.8000, 0.5000]),
			indices=tensor([1, 0, 0])
		)
		
		=> pred.max(1)[1] sẽ là tensor([1, 0, 0])
		
		mà gt = tensor([1, 0, 1])
		(tức là kết quả thực tế lấy từ dataset, kích thước là input là bs x 1) 
		
		sau khi so sánh ==  sẽ ra True/False rồi tiếp theo dùng .float() để chuyển về kiểu float dạng 1. OR 0.
		rồi tính mean() để ra kết quả accuracy
		=> accuracy = 2/3 = 0.6667
	"""

	return (pred.max(1)[1] == gt).float().mean()

# Ref: 
# https://github.com/Sklyvan/Age-Gender-Prediction/blob/main/Age%20%26%20Gender%20Prediction%20Model%20Creation.ipynb
# https://github.com/thepowerfuldeez/age_gender_classifier/blob/master/training.ipynb

class AgeGenderDetectionModel(LightningModule):
	def __init__(self,
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
			output_channels,
			age_classes,
			gender_classes,
		)
		self.loss_functions = {
			# Solution_3
			'Age': nn.CrossEntropyLoss(),
			# 'Age': nn.MSELoss(),
			'Gender': nn.BCEWithLogitsLoss()
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
		age_acc = np.mean(self.age_acc_list)

		# print(f"val epoch {epoch}, gender acc {gender_acc:.2%}, age acc {age_acc:.2%}")
		mean_acc = (gender_acc + age_acc) / 2
		self.log('val_acc', mean_acc, prog_bar=True, on_epoch=True)

		self.gender_acc_list.clear()
		self.age_acc_list.clear()

	def on_test_epoch_start(self) -> None:
		self.gender_acc_list = []
		self.age_acc_list = []
	
	def on_test_epoch_end(self) -> None:
		gender_acc = np.mean(self.gender_acc_list)
		age_acc = np.mean(self.age_acc_list)

		# print(f"val epoch {epoch}, gender acc {gender_acc:.2%}, age acc {age_acc:.2%}")
		mean_acc = (gender_acc + age_acc) / 2
		self.log('test_acc', mean_acc, prog_bar=True, on_epoch=True)

		self.gender_acc_list.clear()
		self.age_acc_list.clear()

	def training_step(self, batch, batch_idx):
		# if len(batch) == 0 : return torch.tensor(0.)
		if len(batch) == 0 : return torch.tensor(0.0, requires_grad=True)

		image, (gender_gt, age_gt) = batch
		gender_logits, age_logits = self(image)
		# print(f"Gender = {gender_gt} => Logits = {gender_logits}")
		# print(f"Age = {age_gt} => Logits = {age_logits}")
		
		# BCE expects one-hot vector
		# 'Gender': nn.BCEWithLogitsLoss()
		gender_gt_onehot = torch.zeros(*gender_logits.size(), device=gender_logits.device)
		gender_gt_onehot = gender_gt_onehot.scatter_(1, gender_gt.unsqueeze(-1).long(), 1)
		gender_loss = self.loss_functions['Gender'](gender_logits, gender_gt_onehot)  # bce
		
		# 'Age': nn.CrossEntropyLoss(),
		age_gt = age_gt.long()
		age_loss = self.loss_functions['Age'](age_logits, age_gt)  # ce
		
		losses = (gender_loss + age_loss) / 2

		self.log('train_acc', 1 - losses, prog_bar=True, on_step=True, on_epoch=True)
		self.log('train_age_acc', 1 - age_loss, prog_bar=True, on_step=True, on_epoch=True)
		self.log('train_gender_acc', 1 - gender_loss, prog_bar=True, on_step=True, on_epoch=True)

		return losses


	def eval_step(self, batch, batch_idx, prefix: str):
		# if random.random() < 0.1:
		if len(batch) == 0: return

		image, (gender_gt, age_gt) = batch

		gender_logits, age_logits = self(image)

		age_acc = accuracy(age_logits, age_gt).item()
		self.log('val_age_acc', age_acc, prog_bar=True, on_step=True, on_epoch=True)
		self.age_acc_list.append(age_acc)

		gender_acc = accuracy(gender_logits, gender_gt).item()
		self.log('val_gender_acc', gender_acc, prog_bar=True, on_step=True, on_epoch=True)
		self.gender_acc_list.append(gender_acc)


	def validation_step(self, batch, batch_idx):
		return self.eval_step(batch, batch_idx, "val")
	
	def test_step(self, batch, batch_idx):
		image, (gender_gt, age_gt) = batch

		# real_age = int(torch.argmax(y[0][:Classes]))
        # read_gender = int(torch.argmax(y[0][Classes:]))

		# implement your own
		gender_logits, age_logits = self(image)
		# print(prediction)
		# print(f'(Test-REAL) gender={AGES[age_gt]} age={"Male" if gender_gt == 0 else "Female"}')
		# pred_gender = int(torch.argmax(prediction[0]))
		# pred_age = int(torch.argmax(prediction[1]))
		# print(f'(Test-PREDICT) gender={AGES[pred_age]} age={"Male" if pred_gender == 0 else "Female"}')

		# plt.title(f'{AGES[pred_age]} {"Male" if pred_gender == 0 else "Female"}')
		# plt.imshow(image, cmap='gray')
		# plt.axis('off')
		# plt.show()

		gender_acc = accuracy(gender_logits, gender_gt).item()
		self.log('test_gender_acc', gender_acc, prog_bar=True, on_step=True, on_epoch=True)
		self.gender_acc_list.append(gender_acc)

		age_acc = accuracy(age_logits, age_gt).item()
		self.log('test_age_acc', age_acc, prog_bar=True, on_step=True, on_epoch=True)
		self.age_acc_list.append(age_acc)

