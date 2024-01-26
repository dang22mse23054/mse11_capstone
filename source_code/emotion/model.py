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
from models.emotion_resnet50 import EmotionResNet50v1, EmotionResNet50v2, EmotionResNet50v3
import matplotlib.pyplot as plt
from common.constants import Constants

# Ref: 
# https://www.kaggle.com/code/tunguyentan/face-emotion-recognition-using-resnet50/notebook

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

class EmotionDetectionModel(LightningModule):
	def __init__(self,
				lr: float = 1e-3,
				momentum: float = 0.9,
				weight_decay: float = 1e-4,
				max_epochs: int = 20,
				**kwargs
	):
		super().__init__()

		# bắt buộc phải truyền param vào __init__ và khai báo self.param_name = param_name 
		# thì khi dùng save_hyperparameters() mới có thể lưu lại được các param này
		# và để sử dụng được các param này thì phải dùng self.hparams.param_name
		self.lr = lr 
		self.momentum = momentum
		self.weight_decay = weight_decay 
		self.max_epochs = max_epochs 

		self.save_hyperparameters()
		
		# self.model = EmotionResNet50v1()
		# self.model = EmotionResNet50v2()
		self.model = EmotionResNet50v3()
		
		self.loss_function = nn.CrossEntropyLoss()
	
	def forward(self, x):
		return self.model(x)

	# version 1
	# def configure_optimizers(self):
	# 	optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay / self.hparams.max_epochs)
	# 	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-7)
	# 	return {
	# 		'optimizer': optimizer,
	# 		'lr_scheduler': {
	# 			'scheduler': scheduler,
	# 			'monitor': 'val_loss',  # Điều chỉnh monitor theo mục bạn muốn
	# 		}
	# 	}
	
	# version 2
	# def configure_optimizers(self):
	# 	optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
	# 	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-4)
	# 	return {
	# 		'optimizer': optimizer,
	# 		'lr_scheduler': {
	# 			'scheduler': scheduler,
	# 			'monitor': 'val_loss',  # Điều chỉnh monitor theo mục bạn muốn
	# 		}
	# 	}
	
	# version 3
	def configure_optimizers(self):
		return optim.Adam(self.parameters(), lr=self.hparams.lr)
	
	def training_step(self, batch, batch_idx):
		# if len(batch) == 0 : return torch.tensor(0.)
		if len(batch) == 0 : return torch.tensor(0.0, requires_grad=True)

		# Label thực tế
		image, emotion_id = batch

		# Label dự đoán
		predicted_emotion_id = self(image)
		
		# Using nn.CrossEntropyLoss(),
		loss = self.loss_function (predicted_emotion_id, emotion_id)  # ce
		
		self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

		return loss

	def on_validation_epoch_start(self):
		self.acc_list = []

	def validation_step(self, batch, batch_idx):
		# if random.random() < 0.1:
		if len(batch) == 0 : return 

		# Label thực tế
		image, emotion_id = batch

		# Label dự đoán
		predicted_emotion_logits = self(image)

		# # Lấy nhãn dự đoán bằng cách chọn class có xác suất cao nhất
		# _, preds = torch.max(predicted_emotion_logits, 1)
		
		# Tính accuracy
		val_acc = accuracy(predicted_emotion_logits, emotion_id).item()
		self.acc_list.append(val_acc)
		
		# # Tính toán loss
		loss = self.loss_function(predicted_emotion_logits, emotion_id)

		self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

	def on_validation_epoch_end(self) -> None:
		final_val_acc = np.mean(self.acc_list)

		self.log('val_acc', final_val_acc, prog_bar=True, on_epoch=True)

		self.acc_list.clear()

	def on_test_epoch_start(self):
		self.acc_list = []

	def test_step(self, batch, batch_idx):
		image, emotion_id = batch

		# implement your own
		predicted_emotion_logits = self(image)

		# Tính accuracy
		val_acc = accuracy(predicted_emotion_logits, emotion_id).item()
		self.acc_list.append(val_acc)

		# Tính toán loss
		loss = self.loss_function(predicted_emotion_logits, emotion_id)

		self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

	def on_test_epoch_end(self) -> None:
		final_val_acc = np.mean(self.acc_list)

		self.log('val_acc', final_val_acc, prog_bar=True, on_epoch=True)

		self.acc_list.clear()