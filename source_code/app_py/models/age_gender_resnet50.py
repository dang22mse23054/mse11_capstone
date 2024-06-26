import torch.nn as nn
import torchvision.models as models
from common.constants import Constants

AGE = Constants.Age()
NUM_OF_AGE_GROUPS = len(AGE.Groups)

class AgeGenderResNet50(nn.Module):
	def __init__(self,
				output_channels: int = 512,
				age_classes: int = NUM_OF_AGE_GROUPS,
				gender_classes: int = 2,
	):
		super().__init__()
		
		# Dùng model resnet50 đã train trước đó, 
		# nhg loại bỏ 1 OR 2 layer cuối cùng (avgpool và fc)
		# tạo thành 1 encoder
		model = models.resnet50(pretrained=True)

		# ở đây là loại bỏ layer cuối cùng
		self.encoder = nn.Sequential(*list(model.children())[:-1])

		# Thay thế layer cuối bằng linear layer mới
		# nhiệm vụ của layer này là duỗi thằng feature map sang vector để tính loss 
		# (vì linear layer thường nằm ở cuối cùng của model)
		self.linear = nn.Linear(model.fc.in_features, output_channels)

		# linear layer có thể nằm sau linear layer hoặc sau conv layer
		# nhằm tái sử dụng linear layer đã được train trước đó (transfer learning)
		# nên ở đây có thể dùng linear layer đã được train trước đó 
		# sau đó dẫn tới linear layer mới để tính loss tương ứng với từng task 
		self.age_head = nn.Linear(output_channels, age_classes, 1)
		self.gender_head = nn.Linear(output_channels, gender_classes, 1)
	
	def forward(self, x):
		# Feature extraction using ResNet50
		features = self.encoder(x)

		# Flatten the features
		features = features.view(features.size(0), -1)

		features = self.linear(features)
		
		age_logits = self.age_head(features)
		gender_logits = self.gender_head(features)

		return gender_logits, age_logits

