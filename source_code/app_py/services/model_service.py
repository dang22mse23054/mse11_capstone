import sys
sys.path.append('../')

import torch
import numpy as np
import albumentations as albu

from PIL import Image
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from datetime import datetime

from emotion.model import EmotionDetectionModel
from age_gender.model import AgeGenderDetectionModel
from face_detection.model import FaceDetectionModel
from common.constants import Constants

MODELS = Constants.Models()
AGE = Constants.Age()
AGES = list(Constants.Age().Groups.keys())

class Timer():
	def __init__(self, label=None):
		self.start_time = None
		self.stop_time = None
		self.label = label
  
	def start(self, label=None):
		self.start_time = datetime.now()
		self.label = label
  
	def stop(self):
		self.stop_time = datetime.now()
		print(f"{self.label}: {(self.stop_time - self.start_time).total_seconds()} seconds")
  
class ModelService():
	def __init__(self, model_type = None):
		self.timer = Timer()
  
		if (model_type is None):
			self.models = self.init_models()
			self.f_model = self.models[MODELS.FACE_MODEL]
			self.ag_model = self.models[MODELS.AGE_GENDER_MODEL]
			self.e_model = self.models[MODELS.EMOTION_MODEL]
		else:
			if model_type == MODELS.FACE_MODEL:
				self.f_model = self.init_single_model(model_type)
			elif model_type == MODELS.AGE_GENDER_MODEL:
				self.ag_model = self.init_single_model(model_type)
			elif model_type == MODELS.EMOTION_MODEL:
				self.e_model = self.init_single_model(model_type)
  
	def init_single_model(self, model_type):
		if model_type == MODELS.FACE_MODEL:
			print('===== Loading model: Face Detection =====')
			checkpoint = torch.load("face_detection/checkpoint/epoch=18-val_map=0.4675.ckpt", map_location=torch.device('cpu'))
			f_model = FaceDetectionModel()
			f_model.load_state_dict(checkpoint['state_dict'])
			# Set Models to Evaluation Mode
			f_model.eval()
			return f_model

		elif model_type == MODELS.AGE_GENDER_MODEL:
			print('===== Loading model: Age_Gender Detection =====')
			checkpoint = torch.load("age_gender/checkpoint/epoch=24-val_acc=0.8411-val_age_acc=0.8013-val_gender_acc=0.8857.ckpt", map_location=torch.device('cpu'))
			ag_model = AgeGenderDetectionModel()
			ag_model.load_state_dict(checkpoint['state_dict'])
			# Set Models to Evaluation Mode
			ag_model.eval()
			return ag_model

		elif model_type == MODELS.EMOTION_MODEL:
			print('===== Loading model: Emotion Detection =====')
			checkpoint = torch.load("emotion/checkpoint/inception-epoch=9-val_acc=0.7520.ckpt", map_location=torch.device('cpu'))
			e_model = EmotionDetectionModel()
			e_model.load_state_dict(checkpoint['state_dict'])
			# Set Models to Evaluation Mode
			e_model.eval()
			return e_model

	def init_models(self):
		return {
			MODELS.FACE_MODEL: self.init_single_model(MODELS.FACE_MODEL), 
			MODELS.AGE_GENDER_MODEL: self.init_single_model(MODELS.AGE_GENDER_MODEL),
			MODELS.EMOTION_MODEL: self.init_single_model(MODELS.EMOTION_MODEL)
		}

	def transform_image_for(self, model_type, input_img):
		gray_input_img = Image.open(input_img).convert('RGB') if type(input_img).__name__ == 'FileStorage' else input_img
		output = None

		# Preprocess the image
		if model_type == MODELS.FACE_MODEL:
			# Preprocess the image using torchvision.transforms
			transformer = transforms.Compose([
				transforms.Grayscale(num_output_channels=1),
				transforms.ToTensor(),
			])
			input_tensor = transformer(gray_input_img)

			# img_size = (224,224)
			# preprocess = albu.Compose([
			# 	# albu.Resize(*(np.array(img_size) * 1.25).astype(int)),
			# 	# albu.CenterCrop(*img_size),
			# 	albu.Normalize(),
			# 	ToTensorV2()
			# ])
			# input_tensor = preprocess(image=np.array(input_image))['image']

			output = input_tensor.unsqueeze(0)  # Add batch dimension

		elif model_type == MODELS.EMOTION_MODEL:
			# for inception
			img_size = (299, 299)

			transformer = albu.Compose([
				# albu.Resize(*img_size),
				albu.Resize(*(np.array(img_size) * 1.25).astype(int)),
				albu.CenterCrop(*img_size),
				albu.Normalize(),
				ToTensorV2()
			])

			input_tensor = transformer(image=np.array(gray_input_img))['image']

			output = input_tensor.unsqueeze(0)  # Add batch dimension

		elif model_type == MODELS.AGE_GENDER_MODEL:
			img_size = (224, 224)
			transformer = albu.Compose([
				albu.Resize(*(np.array(img_size) * 1.25).astype(int)),
				albu.CenterCrop(*img_size),
				albu.Normalize(),
				ToTensorV2()
			])

			input_tensor = transformer(image=np.array(gray_input_img)/255)['image']
			
			output = input_tensor.unsqueeze(0)  # Add batch dimension

		return gray_input_img, output

	def predict_for(self, model_type, input_img):
		self.timer.start(f"({model_type}) transform image")
		gray_input_img, transformed_img_file = self.transform_image_for(model_type, input_img)
		self.timer.stop()

		if model_type == MODELS.FACE_MODEL:
			self.timer.start(f"({model_type}) predict")
			prediction = self.f_model(transformed_img_file)
			self.timer.stop()
   
			prediction = prediction[0]
			scores = prediction['scores'].tolist()
			min_score=0.8
			# face_images = []
			output = []

			for index, bbox in enumerate(prediction['boxes']):
				if scores[index] > min_score:
					bbox = bbox.tolist()
					face = gray_input_img.crop(bbox)
					# face_images.append(face)
					output.append({
						'bbox': bbox,
						'score': scores[index],
						'face_img': face
                    })
					# print(f"Save image face {index}_face.jpg")
					# face.save(f"./img/{index}_face.jpg")
     
			return output
				
		elif model_type == MODELS.EMOTION_MODEL:
			self.e_model(transformed_img_file)


		elif model_type == MODELS.AGE_GENDER_MODEL:
			self.timer.start(f"({model_type}) predict")
			prediction = self.ag_model(transformed_img_file)
			self.timer.stop()

			pred_gender = int(torch.argmax(prediction[0]))
			pred_age = int(torch.argmax(prediction[1]))

			print(f'{AGES[pred_age]} {"Male" if pred_gender == 0 else "Female"}')
			return { 'age_group': pred_age, 'gender': pred_gender}
 



	# def advice_ads(request):

