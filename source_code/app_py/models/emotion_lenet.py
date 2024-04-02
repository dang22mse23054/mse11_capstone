import torch.nn as nn
import torchvision.models as models
from common.constants import Constants

EMOTION = Constants.Emotion()
NUM_OF_EMOTIONS = len(EMOTION.Groups)

class EmotionLeNet(nn.Module):
	def __init__(self,
				emotion_classes: int = NUM_OF_EMOTIONS,
				output_channels: int = 24,
	):
		super().__init__()
		self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0),
			nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0),
			nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
			nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),

        )
		self.classifier = nn.Sequential(
            nn.Linear(128,output_channels),
            nn.ReLU(),
            nn.Linear(output_channels, emotion_classes),
        )
	
	def forward(self, x):
		x = self.feature_extractor(x).squeeze(-1)
		x = self.classifier(x)
		return x