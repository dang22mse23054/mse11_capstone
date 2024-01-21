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
from models.emotion_resnet50 import EmotionResNet50
import matplotlib.pyplot as plt
from common.constants import Constants

# Ref: 
# https://www.kaggle.com/code/tunguyentan/face-emotion-recognition-using-resnet50/notebook

class EmotionDetectionModel(LightningModule):
	def __init__(self,
				lr: float = 1e-3,
				momentum: float = 0.9,
				weight_decay: float = 1e-4,
				**kwargs
	):
		super().__init__()

		# bắt buộc phải truyền param vào __init__ và khai báo self.param_name = param_name 
		# thì khi dùng save_hyperparameters() mới có thể lưu lại được các param này
		# và để sử dụng được các param này thì phải dùng self.hparams.param_name
		self.lr = lr 
		self.momentum = momentum
		self.weight_decay = weight_decay 

		self.save_hyperparameters()
		
		self.model = EmotionResNet50()
		self.loss_function = nn.CrossEntropyLoss()
	
	def forward(self, x):
		return self.model(x)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=0.0001)
		return {
			'optimizer': optimizer,
			'lr_scheduler': {
				'scheduler': scheduler,
				'monitor': 'val_loss',  # Điều chỉnh monitor theo mục bạn muốn
			}
		}
	
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

	def validation_step(self, batch, batch_idx):
		# if random.random() < 0.1:
		if len(batch) == 0 : return 

		# Label thực tế
		image, emotion_id = batch

		# Label dự đoán
		predicted_emotion_id = self(image)
		
		# Tính toán loss
		loss = self.loss_function(predicted_emotion_id, emotion_id)

		self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

	def test_step(self, batch, batch_idx):
		image, emotion_id = batch

		# implement your own
		predicted_emotion_id = self(image)

		# Tính toán loss
		loss = self.loss_function(predicted_emotion_id, emotion_id)

		self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
