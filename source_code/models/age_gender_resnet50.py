import torch.nn as nn
import torchvision.models as models

'''
	num_classes (int): number of output classes of the model (>>> including the Background <<<).
		If box_predictor is specified, num_classes should be None.
		Tức là nếu chỉ nhận biết đâu là Face thì sẽ setting bằng 2 classes (Face, Background)
'''
class AgeGenderResNet50(nn.Module):
	def __init__(self,
				output_channels: int = 512,
				age_classes: int = 5,
				gender_classes: int = 2,
	):
		super().__init__()
		
		# Dùng model resnet50 đã train trước đó, nhg loại bỏ 2 layer cuối cùng (avgpool và fc)
		# tạo thành 1 encoder
		model = models.resnet50(pretrained=True)
		
		self.encoder = nn.Sequential(*list(model.children())[:-1])

		self.linear = nn.Linear(model.fc.in_features, output_channels)
		
		self.age_head = nn.Linear(output_channels, age_classes, 1)
		self.gender_head = nn.Linear(output_channels, gender_classes, 1)

	
	def forward(self, x):
		# Feature extraction using ResNet50
		features = self.encoder(x)
		features = features.view(features.size(0), -1)

		features = self.linear(features)
		
		age_logits = self.age_head(features)
		gender_logits = self.gender_head(features)

		return gender_logits, age_logits

