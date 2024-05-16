import sys
sys.path.append('../')

import cv2
import traceback
import numpy as np
import mediapipe as mp
import time, math
from PIL import Image
from common.constants import Constants
from utils.timer import Timer

COLOR = Constants.COLOR
POSITION = Constants.Position()
mp_face_mesh = mp.solutions.face_mesh

# Utility Functions
def calc_euclidean_distance(point, point1, image):
	img_h, img_w, img_c = image.shape

	x = point.x * img_w
	y = point.y * img_h
	x1 = point1.x * img_w
	y1 = point1.y * img_h

	distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
	return distance

def calc_iris_position(iris_center, right_point, left_point, image = None):
	center_to_right = calc_euclidean_distance(iris_center, right_point, image)
	center_to_left = calc_euclidean_distance(iris_center, left_point, image)
	total = calc_euclidean_distance(right_point, left_point, image)
	ratio = np.round(center_to_right/total, 2)
 
	position = "" 
	
	# left 0.32 <= center <= 0.41 right
	if ratio < 0.44:
		position = POSITION.RIGHT
	elif 0.44 <= ratio and ratio <= 0.62:
		position = POSITION.CENTER
	else:
		position = POSITION.LEFT
	
	return position, ratio

def calc_iris_distance(landmarks, image):
	# === Right eye === #
	# horizontal line 
	rh_right = landmarks[362]
	rh_left = landmarks[263]
	rh_center = landmarks[473]

	return calc_iris_position(rh_center, rh_right, rh_left, image)

# Eyes: close/open
def calc_eyes_distances(landmarks, image=None):
	# === Left eye === #
	# horizontal line 
	lh_right = landmarks[263]
	lh_left = landmarks[362]
	# vertical line 
	lv_top = landmarks[386]
	lv_bottom = landmarks[374]

	# Finding Distance Left Eye
	lhDistance = calc_euclidean_distance(lh_right, lh_left, image)
	lvDistance = calc_euclidean_distance(lv_top, lv_bottom, image)

	# === Right eye === #
 
	# horizontal line 
	rh_right = landmarks[133]
	rh_left = landmarks[33]
	# vertical line 
	rv_top = landmarks[159]
	rv_bottom = landmarks[145]


	# Finding Distance Right Eye
	rhDistance = calc_euclidean_distance(rh_right, rh_left, image)
	rvDistance = calc_euclidean_distance(rv_top, rv_bottom, image)

	# Finding ratio of LEFT and Right Eyes
	reRatio = rhDistance/rvDistance
	leRatio = lhDistance/lvDistance
	ratio = (reRatio+leRatio)/2

	return ratio


class AttentionService():
	def __init__(self):
		self.timer = Timer(is_enabled=False)
		self.face_mesh = mp_face_mesh.FaceMesh(
			max_num_faces=1,
			refine_landmarks=True,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5
		)

	def predict(self, image):

		# WARNNG: NOT reuse face_mesh because wrong calc when using inside the loop 
		# face_mesh = mp_face_mesh.FaceMesh(
		# 	max_num_faces=1,
		# 	refine_landmarks=True,
		# 	min_detection_confidence=0.5,
		# 	min_tracking_confidence=0.5
		# )
		try:
			
			
			self.timer.start(f"(attention) landmarks detection")

			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			image.flags.writeable = False
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			# https://www.youtube.com/watch?v=-toNMaS4SeQ
			img_h, img_w, img_c = image.shape
			face_3d = []
			face_2d = []

			results = self.face_mesh.process(image)
			
			# Convert the BGR image to RGB before processing.
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			self.timer.stop()
			
			# Print and draw face mesh landmarks on the image.
   
			if results.multi_face_landmarks:
				self.timer.start(f"(attention) calc attention")
				face_attention = []
				for face_landmarks in results.multi_face_landmarks:
					landmarks = face_landmarks.landmark
					
					for idx, lm in enumerate(face_landmarks.landmark):
						if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
							# tìm tọa độ thực tế của các điểm landmark
							# vì tọa độ trả về từ model là tọa độ dạng % nên cần nhân với chiều rộng và chiều cao của ảnh
							cx, cy = int(lm.x * img_w), int(lm.y * img_h)

							# get the 2D/3D coordinates
							face_2d.append([cx, cy])
							face_3d.append([cx, cy, lm.z])
						
					# convert to numpy array
					face_2d = np.array(face_2d, dtype=np.float64)
					face_3d = np.array(face_3d, dtype=np.float64)

					# the camera matrix
					# chỗ này chưa hiểu lắm
					focal_length = 1 * img_w

					cam_matrix = np.array([ [focal_length, 0, img_h//2],
											[0, focal_length, img_w//2],
											[0, 0, 1]]
											# , dtype=np.float64
										)
					
					# the distortion matrix
					dist_matrix = np.zeros((4,1), dtype=np.float64)

					# solve PnP
					# success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
					success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

					# get rotation matrix
					rotation_matrix, f_jac = cv2.Rodrigues(rotation_vector)

					# get angles
					angles, f_mtxR, f_mtxQ, f_Qx, f_Qy, f_Qz = cv2.RQDecomp3x3(rotation_matrix)
					# get the y rotation degree
					x = angles[0] * 35
					y = angles[1] * 35
					z = angles[2] * 35
					
					face_position = None
					if y < -3.7:
						face_position = POSITION.RIGHT
					elif y > 3.7:
						face_position = POSITION.LEFT
					elif x < -1.2:
						face_position = POSITION.DOWN
					elif x > 4:
						face_position = POSITION.UP
					else:
						face_position = POSITION.CENTER

					txt = f'Nose: x={str(np.round(x, 2)):4} y={str(np.round(y, 2)):4} z={str(np.round(z, 2)):4}'
					print(txt + ' => ' + POSITION.Label[face_position])

					# tính toán tỉ lệ mắt để xác định trạng thái mắt đang nhắm hay mở
					# thêm image vào để vẽ các đường line (if necessary)
					eyes_ratio = calc_eyes_distances(landmarks, image)
					
					open_eyes = True
					if eyes_ratio > 5.3:
						open_eyes = False

					iris_position, iris_ratio = calc_iris_distance(landmarks, image)

					is_attention = False
					if open_eyes == True:
						if face_position == POSITION.CENTER and iris_position == POSITION.CENTER:
							is_attention = True
						elif face_position == POSITION.LEFT and iris_position == POSITION.RIGHT:
							is_attention = True
						elif face_position == POSITION.RIGHT and iris_position == POSITION.LEFT:
							is_attention = True
						
					face_attention.append({
						'is_attention': is_attention,

						'detail': {
							'open_eyes': open_eyes,
							'face_position': face_position,
							'iris_position': iris_position,
							'eyes_ratio': eyes_ratio,
							'face_direction': {'x': x, 'y': y, 'z': z},
							'iris_ratio': iris_ratio,
						},
					})
				self.timer.stop()

				return face_attention
		except Exception as error:
			print(traceback.format_exc())

		return None