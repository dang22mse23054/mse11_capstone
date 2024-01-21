import torch.nn as nn
import torchvision.models as models
from common.constants import Constants

EMOTION = Constants.Emotion()
NUM_OF_EMOTIONS = len(EMOTION.Groups)

class EmotionResNet50(nn.Module):
	def __init__(self,
				emotion_classes: int = NUM_OF_EMOTIONS,
				output_channels: int = 32,
	):
		super().__init__()
		
		# Dùng model resnet50 đã train trước đó, 
		# nhg loại bỏ 1 OR 2 layer cuối cùng (avgpool và fc)
		self.base_model = models.resnet50(pretrained=True)

		# ở đây là loại bỏ layer cuối cùng
		# giống như self.encoder = nn.Sequential(*list(model.children())[:-1]) của age_gender_resnet50
		# lớp Identity này không làm gì cả, chỉ lấy input và trả về chính nó
		self.base_model.fc = nn.Identity()

		# bắt đầu từ đây, nối tiếp nhiều layer mới
		self.dropout = nn.Dropout(0.5)
		self.flatten = nn.Flatten()

		# ở đây là thêm 1 layer mới, 
		# nn.BatchNorm1d(2048) để chuẩn hóa lại dữ liệu đầu ra của layer trước đó 
		self.batch_norm1 = nn.BatchNorm1d(2048)
		# nn.Linear(2048, 32) để giảm số chiều của dữ liệu đầu ra của layer trước đó từ 2048 xuống 32
		# vì ta chỉ cần 32 features để phân loại emotion
		self.fc1 = nn.Linear(2048, output_channels)

		# tiếp tục BatchNorm1d 
		self.batch_norm2 = nn.BatchNorm1d(output_channels)
		self.relu = nn.ReLU()

		# self.fc2 = nn.Linear(output_channels, output_channels)
		# self.batch_norm3 = nn.BatchNorm1d(output_channels)

		# self.fc3 = nn.Linear(output_channels, output_channels)
		# self.batch_norm4 = nn.BatchNorm1d(output_channels)

		self.output = nn.Linear(output_channels, emotion_classes)

		# khi dùng CrossEntropyLoss() thì ko cần dùng softmax ở trong model nữa
		# self.softmax = nn.Softmax(dim=1)
	
	def forward(self, x):
		# Feature extraction using ResNet50
		x = self.base_model(x)
		x = self.dropout(x)

		# Flatten the features for using in Linear layers
		x = self.flatten(x)
		x = self.batch_norm1(x)

		x = self.fc1(x)
		x = self.batch_norm2(x)
		
		# x = self.relu(x)
		# x = self.dropout(x)
		# x = self.fc2(x)
		# x = self.batch_norm3(x)
		
		# x = self.relu(x)
		# x = self.dropout(x)
		# x = self.fc3(x)
		# x = self.batch_norm4(x)
		
		x = self.relu(x)
		x = self.output(x)

		# khi dùng CrossEntropyLoss() thì ko cần dùng softmax ở trong model nữa
		# x = self.softmax(x)
		return x

