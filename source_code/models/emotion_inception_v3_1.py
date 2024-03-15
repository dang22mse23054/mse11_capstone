import torch.nn as nn
import torchvision.models as models
from common.constants import Constants

EMOTION = Constants.Emotion()
NUM_OF_EMOTIONS = len(EMOTION.Groups)

class EmotionInceptionV3_1(nn.Module):
	def __init__(self,
				emotion_classes: int = NUM_OF_EMOTIONS,
	):
		super().__init__()
		
		# Dùng model inception_v3 đã train trước đó, 
		self.base_model = models.inception_v3(pretrained=True)
		# nhg loại bỏ aux_logits layer 
		self.base_model.aux_logits = False

		num_ftrs = self.base_model.fc.in_features
		# lớp Identity này không làm gì cả, chỉ lấy input và trả về chính nó
		self.base_model.fc = nn.Linear(num_ftrs, emotion_classes)

	
	def forward(self, x):
		return self.base_model(x)
