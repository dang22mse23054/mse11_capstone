import cv2
import numpy as np
import mediapipe as mp
import time, math
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

from services.model_service import Constants, ModelService
model_service = ModelService(Constants.Models.FACE_MODEL)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

COLOR = {
	"GREEN": (0, 255, 0),
	"RED": (0, 0, 255),
	"BLUE": (255, 0, 0),
	"YELLOW": (0, 255, 255),
	"PURPLE": (255, 0, 255),
}

# ============================================================= #
def drawPoint(image, point, label = 'unknown'):
	cv2.putText(image, str(label), point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR['GREEN'], 1)

# Euclaidean distance
def euclaideanDistance(point, point1, image):
	img_h, img_w, img_c = image.shape

	x = point.x * img_w
	y = point.y * img_h
	x1 = point1.x * img_w
	y1 = point1.y * img_h

	distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
	return distance

def irisPosition(iris_center, right_point, left_point, image = None):
	center_to_right = euclaideanDistance(iris_center, right_point, image)
	center_to_left = euclaideanDistance(iris_center, left_point, image)
	total = euclaideanDistance(right_point, left_point, image)
	ratio = np.round(center_to_right/total, 2)

	position = "" 
	
	# left 0.32 <= center <= 0.41 right
	if ratio < 0.44:
		position = "RIGHT"
		color = COLOR['BLUE']
	elif 0.44 <= ratio and ratio <= 0.62:
		position = "CENTER" 
		color = COLOR['GREEN']
	else:
		position = "LEFT" 
		color = COLOR['RED']
	
	# logging
	info = f"Iris: center_to_right={np.round(center_to_right, 2):4}, center_to_left={np.round(center_to_left, 2):4}, total={np.round(total, 2):4}, ratio={np.round(ratio, 2):4} => {position}"
	if image is not None:
		cv2.putText(image, info, (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	else:
		print(info)

	return position, ratio, color

def irisDistance(landmarks, image):
	img_h, img_w, img_c = image.shape

	# === Right eye === #
	# horizontal line 
	rh_right = landmarks[362]
	rh_left = landmarks[263]
	rh_center = landmarks[473]

	# === Iris === #
	# lấy tọa độ tâm mắt (Iris - con ngươi) -> sau đó vẽ ra
	right_iris = (int(landmarks[473].x * img_w), int(landmarks[473].y * img_h))
	cv2.circle(image, right_iris, radius=2, color=(0,255,255), thickness=2)
	left_iris = (int(landmarks[468].x * img_w), int(landmarks[468].y * img_h))
	cv2.circle(image, left_iris, radius=2, color=(0,255,255), thickness=2)

	return irisPosition(rh_center, rh_right, rh_left, image)


def eyesDistances(landmarks, image=None):
	# === Left eye === #
	# horizontal line 
	lh_right = landmarks[263]
	lh_left = landmarks[362]
	# vertical line 
	lv_top = landmarks[386]
	lv_bottom = landmarks[374]

	# Finding Distance Left Eye
	lhDistance = euclaideanDistance(lh_right, lh_left, image)
	lvDistance = euclaideanDistance(lv_top, lv_bottom, image)

	# === Right eye === #
 
	# horizontal line 
	rh_right = landmarks[133]
	rh_left = landmarks[33]
	# vertical line 
	rv_top = landmarks[159]
	rv_bottom = landmarks[145]


	# Finding Distance Right Eye
	rhDistance = euclaideanDistance(rh_right, rh_left, image)
	rvDistance = euclaideanDistance(rv_top, rv_bottom, image)

	# Finding ratio of LEFT and Right Eyes
	reRatio = rhDistance/rvDistance
	leRatio = lhDistance/lvDistance
	ratio = (reRatio+leRatio)/2


	# draw lines on right eyes 
	# if image is not None:
	# 	img_h, img_w, img_c = image.shape

	# 	# vẽ các đường giữa top-bottom, left-right của mắt 
  	# 	# -- vẽ đường ngang (left-right)
	# 	eye_pLeft = (int(landmarks[362].x * img_w), int(landmarks[362].y * img_h))
	# 	drawPoint(image, eye_pLeft, label = '362')
	# 	eye_pRight = (int(landmarks[263].x * img_w), int(landmarks[263].y * img_h))
	# 	drawPoint(image, eye_pRight, label = '263')
		
	# 	cv2.line(image, eye_pLeft, eye_pRight, (255, 255,0), 2)
	# 	# hiện index của tọa độ các điểm landmark cần biết của 

	# 	# -- vẽ đường dọc (top-bottom)
	# 	eye_pTop = (int(landmarks[386].x * img_w), int(landmarks[386].y * img_h))
	# 	drawPoint(image, eye_pTop, label = '386')
	# 	eye_pBottom = (int(landmarks[374].x * img_w), int(landmarks[374].y * img_h))
	# 	drawPoint(image, eye_pBottom, label = '374')
		
	# 	cv2.line(image, eye_pTop, eye_pBottom, (255, 255,0), 2)

	return ratio

# ============================================================= #
cap = None
screen_name = '(MediaPipe) Head_pose & Eyes_tracker'
cv2.namedWindow(screen_name)        # Create a named window
cv2.moveWindow(screen_name, 60,30)  # Move it to (60,30)

face_detection = mp_face_detection.FaceDetection(
	model_selection=1,
	min_detection_confidence=0.95
)

# WARNNG: NOT reuse face_mesh because wrong calc when using inside the loop 
# face_mesh = mp_face_mesh.FaceMesh(
# 	max_num_faces=1,
# 	refine_landmarks=True,
# 	min_detection_confidence=0.5,
# 	min_tracking_confidence=0.5
# )

# # Dùng Camera (CAEMRA) ===== BEGIN
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
# 	success, image = cap.read()
# 	if not success:
# 		print("Ignoring empty camera frame.")
# 		# If loading a video, use 'break' instead of 'continue'.
# 		continue
# # Dùng Camera (CAEMRA) ===== END

# Dùng TEST folder (TEST_IMAGE) ===== BEGIN
PATHS = {
	'img_list': 'face_detection/raw/wider_face_split/wider_face_testdemo_filelist.txt',
	'img_dir': 'face_detection/raw/WIDER_test/images',
}
img_list_file = PATHS["img_list"]
# Read the list of image files
with open(img_list_file, 'r') as file:
	file_list = [line.strip() for line in file.readlines()]

for file_path in file_list:
	file_path = f'{PATHS["img_dir"]}/{file_path}'
	ori_img = Image.open(file_path)
# Dùng TEST folder (TEST_IMAGE) ===== END

	# # == Cách 1: dùng model tự train
	# # NOTE: comment this line if using TEST_IMAGE
	# pil_image = Image.fromarray(np.uint8(ori_img)).convert('RGB')
	# output_faces = model_service.predict_for(Constants.Models.FACE_MODEL, pil_image)
	# if output_faces:
	# 	for ind, detection in enumerate(output_faces):
	# 		# Dạng PIL Image
	# 		image = detection['face_img']
	# 		# chuyển sang dạng numpy array để face mesh đọc dc
	# 		image = np.array(image, dtype=np.uint8)
	# # ============== Hết cách 1 =============
	
	# == Cách 2: dùng model FaceDetection của Mediapipe
	# chuyển sang dạng numpy array để face mesh đọc dc
	# NOTE: uncomment this line if using TEST_IMAGE
	image = np.array(ori_img, dtype=np.uint8)
	
	# Perform face detection
	image.flags.writeable = False
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	face_results = face_detection.process(image)

	image.flags.writeable = True
	# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	
	ih, iw, _ = image.shape
	origin_image = image

	if face_results.detections:
		for detection in face_results.detections:
			
			bboxC = detection.location_data.relative_bounding_box
			x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
			cv2.rectangle(origin_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

			# Perform pose estimation using the face region
			# face_center = (x + w // 2, y + h // 2)

			# frame chứa khuôn mặt thôi
			image = origin_image[y:y + h, x:x + w]
	# ============== Hết cách 2 =============

			start = time.time()

			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			image.flags.writeable = False
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			# https://www.youtube.com/watch?v=-toNMaS4SeQ
			img_h, img_w, img_c = image.shape
			face_3d = []
			face_2d = []

			face_mesh = mp_face_mesh.FaceMesh(
				max_num_faces=1,
				refine_landmarks=True,
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5
			) 
			results = face_mesh.process(image)

			# Convert the BGR image to RGB before processing.
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			# Print and draw face mesh landmarks on the image.
			if results.multi_face_landmarks:
				for face_landmarks in results.multi_face_landmarks:
					landmarks = face_landmarks.landmark
					
					for idx, lm in enumerate(face_landmarks.landmark):
						if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
							if idx == 1:
								nose_2d = (lm.x * img_w, lm.y * img_h)
								nose_3d = (lm.x * img_w, lm.y * img_h, lm.z)
								# nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

							# tìm tọa độ thực tế của các điểm landmark
							# vì tọa độ trả về từ model là tọa độ dạng % nên cần nhân với chiều rộng và chiều cao của ảnh
							cx, cy = int(lm.x * img_w), int(lm.y * img_h)

							# get the 2D/3D coordinates
							face_2d.append([cx, cy])
							face_3d.append([cx, cy, lm.z])

							# hiện index của tọa độ các điểm landmark cần biết của 
							drawPoint(image, (cx, cy), label = str(idx))
				
					# convert to numpy array
					face_2d = np.array(face_2d, dtype=np.float64)
					face_3d = np.array(face_3d, dtype=np.float64)

					# the camera matrix
					# chỗ này chưa hiểu lắm
					focal_length = 1 * img_w

					# camera matrix chính là ma trận chiếu hình học của camera
					# nó bao gồm focal length, principal point, image size, distortion coefficients
					# focal length là khoảng cách từ tâm camera đến màn hình
					# img_h/2, img_w/2 là tọa độ của trục x, y của ảnh
					# việc chia 2 là để lấy tâm của ảnh
					# số 1 ở cuối là scale factor (đơn vị pixel) chính là độ dài của 1 pixel
					cam_matrix = np.array([ [focal_length, 0, img_h//2],
											[0, focal_length, img_w//2],
											[0, 0, 1]]
											# , dtype=np.float64
										)
					
					# the distortion matrix
					dist_matrix = np.zeros((4,1), dtype=np.float64)

					# solve PnP
					# pnp là phương pháp giải quyết vấn đề 3D-2D dùng để xác định vị trí và hướng của một đối tượng trong không gian 3D từ hình ảnh 2D
					# cụ thể là xác định vị trí và hướng của khuôn mặt từ hình ảnh 2D
					# bằng cách sử dụng các điểm landmark của khuôn mặt và các thông số như camera matrix, distortion matrix 
					# success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
					success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
					
					# kết quả đầu ra là rotation_vector và translation_vector
					# rotation_vector là vector quay của khuôn mặt
					# translation_vector là vector dịch chuyển của khuôn mặt
		
					# thông qua rotation vector, ta có thể tìm ra các góc quay của khuôn mặt
					# get rotation matrix
					rotation_matrix, jac = cv2.Rodrigues(rotation_vector)

					# kết quả có mtxR, mtxQ, Qx, Qy, Qz là các thông số khác của rotation matrix
					# cụ thể là các ma trận quay, quay quanh các trục x, y, z
					# trong đó Qx, Qy, Qz là các góc quay quanh các trục x, y, z
					# mtx là ma trận quay dùng để quay vật thể từ hệ tọa độ thế giới sang hệ tọa độ camera
					# mtxQ là ma trận quay dùng để quay vật thể từ hệ tọa độ camera sang hệ tọa độ thế giới
					# angles là góc quay của vật thể quanh các trục x, y, z
					# get angles
					angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)

					# get the y rotation degree
					# x = angles[0] * 360
					# y = angles[1] * 360
					# z = angles[2] * 360
					x = angles[0] * 30
					y = angles[1] * 30
					z = angles[2] * 30

					# từ các góc x y z, ta có thể xác định hướng của khuôn mặt
					# cụ thể là hướng của mũi

					color = COLOR['GREEN']
					if y < -3.7:
						text = "RIGHT"
						color = COLOR['BLUE']
					elif y > 3.7:
						text = "LEFT"
						color = COLOR['RED']
					elif x < -1.2:
						text = "DOWN"
						color = COLOR['PURPLE']
					elif x > 4:
						text = "UP"
						color = COLOR['YELLOW']
					else:
						text = "Forward"


					# display the nose direction
					# nose_3d_projected, jacobian = cv2.projectPoints(nose_3d, rotation_vector, translation_vector, cam_matrix, dist_matrix)
					
					p1 = (int(nose_2d[0]), int(nose_2d[1]))
					p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

					cv2.line(image, p1, p2, COLOR['BLUE'], 3)

					# Add the text on the image
					cv2.putText(image, f'Face: {text}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
					txt = f'Nose: x={str(np.round(x, 2)):4} y={str(np.round(y, 2)):4} z={str(np.round(z, 2)):4}'
					print(txt + ' => ' + text)
					cv2.putText(image, txt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR['BLUE'], 2)

					end = time.time()
					total_time = end - start

					fps = 1/total_time
					cv2.putText(image, f'FPS: {str(np.round(fps, 2))}', (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR['RED'], 2)

					# tính toán tỉ lệ mắt để xác định trạng thái mắt đang nhắm hay mở
					# thêm image vào để vẽ các đường line (if necessary)
					eyes_ratio = eyesDistances(landmarks, image)
					cv2.putText(image, f'eyes_ratio: {str(np.round(eyes_ratio, 2))}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR['RED'], 2)

					if eyes_ratio > 5.3:
						cv2.putText(image, "Nham mat - Blink", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR['YELLOW'], 2)


					iris_position, iris_ratio, iris_color = irisDistance(landmarks, image)
					print(f'>> Iris: {iris_position} (ratio: {str(np.round(iris_ratio, 2))})')
					cv2.putText(image, f'Iris: {iris_position}', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, iris_color, 3)
					cv2.putText(image, f'(ratio: {str(np.round(iris_ratio, 2))})', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR['BLUE'], 2)
					


					# dùng tool có sẵn của lib để vẽ các connection giữa các điểm landmark
					# mp_drawing.draw_landmarks(
					# 	image=image,
					# 	landmark_list=face_landmarks,
					# 	connections=mp_face_mesh.FACEMESH_CONTOURS,
					# 	landmark_drawing_spec=drawing_spec,
					# 	connection_drawing_spec=drawing_spec
					# )

				# Dùng TEST folder (TEST_IMAGE) ===== BEGIN
				# Flip the image horizontally for a selfie-view display.
				# cv2.imshow(screen_name, cv2.flip(image, 1))
				cv2.imshow(screen_name, image)
				if cv2.waitKey(1500) & 0xFF == 27:
					stop = True
				# Dùng TEST folder (TEST_IMAGE) ===== END
				
	
		# # Dùng Camera (CAEMRA) ===== BEGIN
		# # Flip the image horizontally for a selfie-view display.
		# # cv2.imshow(screen_name, cv2.flip(image, 1))
		# cv2.imshow(screen_name, image)
		# if cv2.waitKey(25) & 0xFF == 27:
		# 	break
		# # Dùng Camera (CAEMRA) ===== END
if cap:
	cap.release()