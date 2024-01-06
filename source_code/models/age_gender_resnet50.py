import torch.nn as nn
import torchvision.models as models

'''
	num_classes (int): number of output classes of the model (>>> including the Background <<<).
		If box_predictor is specified, num_classes should be None.
		Tức là nếu chỉ nhận biết đâu là Face thì sẽ setting bằng 2 classes (Face, Background)
'''
class AgeGenderResNet50(nn.Module):
	def __init__(self,
				encoder_channels: int = 2048,
				output_channels: int = 512,
				age_classes: int = 5,
				gender_classes: int = 2,
	):
		super().__init__()
		
		# Dùng model resnet50 đã train trước đó, nhg loại bỏ 2 layer cuối cùng (avgpool và fc)
		# tạo thành 1 encoder
		self.encoder = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])

		# sqeeze-excite là 1 kiến trúc để tăng cường đặc trưng của 1 layer (tăng cường đặc trưng của encoder)
		# https://arxiv.org/pdf/1709.01507.pdf
		self.downsample = nn.Conv2d(encoder_channels, output_channels, 1)
		self.relu = nn.ReLU()
		
		self.age_head = nn.Conv2d(output_channels, age_classes, 1)
		self.gender_head = nn.Conv2d(output_channels, gender_classes, 1)

	
	def forward(self, x):
		# Feature extraction using ResNet50
		features = self.encoder(x)
		features = features.view(features.size(0), -1)

		features = self.downsample(features)
		features = self.relu(features)
		
		age_logits = self.age_head(features)
		gender_logits = self.gender_head(features)

		return gender_logits, age_logits

