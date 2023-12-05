import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
from albumentations import Compose, RandomCrop, BboxParams
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from constants import Constants

wider_face_root = '/kaggle/input/wider-face-a-face-detection-dataset'

COL_NAME = Constants.DFColumns()
MODE = Constants.Mode()
PATHS = {
	MODE.DEMO: {
		'annotation': 'raw/wider_face_split/wider_face_demo_bbx_gt.txt',
		'img_dir': 'raw/WIDER_train/images',
	},
	MODE.VALDEMO: {
		'annotation': 'raw/wider_face_split/wider_face_valdemo_bbx_gt.txt',
		'img_dir': 'raw/WIDER_val/images',
	},
	MODE.TRAIN: {
		'annotation': f'{wider_face_root}/wider_face_annotations/wider_face_split/wider_face_train_bbx_gt.txt',
		'img_dir': f'{wider_face_root}/WIDER_train/WIDER_train/images',
	},
	MODE.VALIDATE: {
		'annotation': f'{wider_face_root}/wider_face_annotations/wider_face_split/wider_face_val_bbx_gt.txt',
		'img_dir': f'{wider_face_root}/WIDER_val/WIDER_val/images',
	},
	MODE.TEST:  {
		'img_dir': 'raw/WIDER_test/images',
	}
}

def show_img(image):
	# Đọc hình ảnh từ file

	# Chuyển đổi hình ảnh từ BGR sang RGB (OpenCV sử dụng BGR, trong khi Matplotlib sử dụng RGB)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Hiển thị hình ảnh
	plt.imshow(image_rgb)
	plt.axis('off')  # Tắt trục
	plt.show()

def get_df(dir):
	
	# Đọc dữ liệu từ file
	with open(dir, 'r') as file:
		lines = file.readlines()

	df = pd.DataFrame(columns=['path', 'detail'])

	total_lines = len(lines)
	print(f"total lines = {total_lines}")
	data = {'path': [], 'detail': []}
	with tqdm(total=total_lines) as pbar:

		# read line
		line_num = 0

		# Lặp qua từng dòng trong lines
		while line_num < len(lines):
			pbar.update(1)

			# Lấy path từ dòng hiện tại
			path = lines[line_num].strip()
			
			# Di chuyển đến dòng chứa số lượng dữ liệu cần đọc
			line_num += 1
			pbar.update(1)

			num_bbox = int(lines[line_num].strip())

			bbox_from_line = line_num + 1
			bbox_to_line = bbox_from_line + (1 if num_bbox == 0 else num_bbox)
			bboxs = lines[bbox_from_line : bbox_to_line]

			# bboxs = list(map(lambda x: x.rstrip().split()[:4], bboxs))
			bboxs = list(map(lambda x: x.rstrip().split(), bboxs))

			pbar.update(bbox_to_line - line_num - 1)
			line_num = bbox_to_line

			data['path'].append(path)
			data['detail'].append(bboxs)

	return pd.DataFrame(data)

'''
Number of bounding box
x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose

Easy image -> all these are 0s
blur, expression, illumination, invalid, occlusion, pose
'''
class ImageDetectionDataset(Dataset):
	def __init__(self,
				 mode : str = MODE.DEMO,
				 transforms: Compose = None):
		"""
		Prepare data for image detection.

		Args:
			dataframe: dataframe with image id and bboxes
			image_dir: path to images
			transforms: albumentations
		"""
		self.mode = mode
		self.df = get_df(PATHS[mode]['annotation'])
		self.image_dir = PATHS[mode]['img_dir']
		self.transforms = transforms

		# print(self.df)

	def __getitem__(self, idx: int):
		# get image info (path, detail)
		img_info = self.df.iloc[idx]

		# print(f"------------ GET ITEM -------- {idx} ---- {img_info.path}")
		image = cv2.imread(f'{self.image_dir}/{img_info.path}', cv2.IMREAD_COLOR).astype(np.float32)

		# normalization.
		image /= 255.0

		target = {}

		MIN_SIZE = 10

		# for train and valid 
		if self.mode != MODE.TEST:
			rows = img_info['detail']

			# convert each item of cols to int
			rows = [[int(value) for value in cols] for cols in rows]
			boxes = [cols[0:4]for cols in rows]
			labels = [1] * len(boxes)

			# convert [x, y, w, h] to [x1, y1, x2, y2]
			boxes = [(x, y, x+w, y+h) for x, y, w, h in boxes]
			
			# filter small boxes
			selected_boxes = [id for id, box in enumerate(boxes) if (box[2] - box[0] >= MIN_SIZE and box[3] - box[1] >= MIN_SIZE)
							and box[2] < image.shape[1] and box[3]  < image.shape[0]]

			boxes = [boxes[id] for id in selected_boxes]
			labels = [labels[id] for id in selected_boxes]

			# set default label 'Face' (value = 1) to each boxes

			target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
			target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
			target['image_id'] = torch.tensor([idx])

			if self.transforms:
				image_dict = {
					'image': image,
					'bboxes': [list(box) + [label] for box, label in zip(boxes, labels)],
					'labels': labels
				}
				image_dict = self.transforms(**image_dict)
				image = image_dict['image']

				boxes = [bbox[:4] for bbox in image_dict['bboxes']]
				labels = [bbox[4] for bbox in image_dict['bboxes']]

				# filter small boxes
				selected_boxes = [id for id, box in enumerate(boxes) if (box[2] - box[0] >= MIN_SIZE and box[3] - box[1] >= MIN_SIZE)
								and box[2] < image.shape[1] and box[3]  < image.shape[0]]

				boxes = [boxes[id] for id in selected_boxes]
				labels = [labels[id] for id in selected_boxes]


				target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
				target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

			if len(boxes) > 0:
				boxes = np.array(boxes)
				area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
				iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
				target['area'] = torch.from_numpy(area)
				target['iscrowd'] = iscrowd

		image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)

		return image, target

	def __len__(self) -> int:
		return len(self.df)