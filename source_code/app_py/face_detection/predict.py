import sys
sys.path.append('../')

import torch
from PIL import Image
from model import FaceDetectionModel
from torchvision import transforms
from PIL import ImageDraw
from common.constants import Constants
from dataset import PATHS


MODE = Constants.Mode()

def init_model():
	checkpoint = torch.load("checkpoint/epoch=9-val_map=0.4393.ckpt", map_location=torch.device('cpu'))
	model = FaceDetectionModel()
	model.load_state_dict(checkpoint['state_dict'])
	
	# Set Model to Evaluation Mode
	model.eval()
	return model

def predict_all(file_list, model):
	predictions = []
	
	with torch.no_grad():
		for file_path in file_list:
			file_path = f'{PATHS[MODE.TESTDEMO]["img_dir"]}/{file_path}'
			input_image = Image.open(file_path)

			# Preprocess the image using torchvision.transforms
			preprocess = transforms.Compose([
				# không hiểu sao dùng cái này thì performance tốt lên 
				transforms.Grayscale(num_output_channels=1),
				transforms.ToTensor(),
			])
			input_tensor = preprocess(input_image)

			input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
			
			output = model(input_batch)

			predictions.append(output)
	
	return predictions

def convert_img_to_features(images, model):
	if len(images) == 0:
		return None
	
	model = model.model
	features = []
	with torch.no_grad():
		images, _= model.transform(images, None)
		features = model.backbone(images.tensors)
	
	print('-------convert_img_to_features-------')
	print(features.keys())
	return features['0']

# số cũ min_score = 0.29
def show_bbox(image, prediction, min_score=0.8):
	image_with_bbox = image.copy()
	draw = ImageDraw.Draw(image_with_bbox)
	prediction = prediction[0]
	scores = prediction['scores'].tolist()

	for index, bbox in enumerate(prediction['boxes']):
		if scores[index] > min_score:
			draw.rectangle(bbox.tolist(), outline="red", width=1)
	image_with_bbox.show()

def extract_faces(image, prediction):
	output_images = []

	# Define a transform to convert the image to tensor
	transform = transforms.ToTensor()

	image = image.copy()
	prediction = prediction[0]
	scores = prediction['scores'].tolist()

	for index, bbox in enumerate(prediction['boxes']):
		if scores[index] > 0.3:
			face = image.crop(bbox.tolist())

			# Convert the image to PyTorch tensor
			face = transform(face)

			output_images.append(face)
			# crop.show()
			
	return output_images

if __name__ == "__main__":
	img_list_file = PATHS[MODE.TESTDEMO]["img_list"]
	model = init_model()
	
	# Read the list of image files
	with open(img_list_file, 'r') as file:
		file_list = [line.strip() for line in file.readlines()]
	
	# Make predictions for all files in the list
	all_predictions = predict_all(file_list, model)
	
	# Get the predictions images
	for file_path, prediction in zip(file_list, all_predictions):
		file_path = f'{PATHS[MODE.TESTDEMO]["img_dir"]}/{file_path}'
		input_image = Image.open(file_path).convert('RGB')
		show_bbox(input_image, prediction)
		# face_images = extract_faces(input_image, prediction)
		# get the face features
		
		print('---------- FACE features ----------')
		features = convert_img_to_features([transforms.ToTensor()(input_image)], model)
	
	