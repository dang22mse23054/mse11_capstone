import cv2
import numpy as np
import mediapipe as mp
import time, math
from PIL import Image
from services.model_service import Constants, ModelService

model_service = ModelService()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

COLOR = {
	"RED": (0, 0, 255),
	"BLUE": (255, 0, 0),
	"GREEN": (0, 255, 0),
	"YELLOW": (0, 255, 255),
	"PURPLE": (255, 0, 255),
}

# ============================================================= #

def drawPoint(image, point, label = 'unknown'):
	cv2.putText(image, str(label), point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR['GREEN'], 1)

def txt(txt):
	print(txt)
	return txt

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
	print(info)

	return position, ratio, color

def irisDistance(landmarks, image):

	# === Right eye === #
	# horizontal line 
	rh_right = landmarks[362]
	rh_left = landmarks[263]
	rh_center = landmarks[473]

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

	return ratio

def detect_attention(image):
	start = time.time()
	
	# To improve performance, optionally mark the image as not writeable to
	# pass by reference.
	# image.flags.writeable = False
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# https://www.youtube.com/watch?v=-toNMaS4SeQ
	img_h, img_w, img_c = image.shape
	face_3d = []
	face_2d = []
	
	eye_2d = []
	eye_3d = []

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
						nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

					# tìm tọa độ thực tế của các điểm landmark
					# vì tọa độ trả về từ model là tọa độ dạng % nên cần nhân với chiều rộng và chiều cao của ảnh
					cx, cy = int(lm.x * img_w), int(lm.y * img_h)

					# get the 2D/3D coordinates
					face_2d.append([cx, cy])
					face_3d.append([cx, cy, lm.z])

					# hiện index của tọa độ các điểm landmark cần biết của 
					# drawPoint(image, (cx, cy), label = str(idx))
				
				if idx == 473 or idx == 263 or idx == 362 or idx == 374 or idx == 386:
					if idx == 473:
						iris_2d = (lm.x * img_w, lm.y * img_h)
						iris_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

					# tìm tọa độ thực tế của các điểm landmark
					# vì tọa độ trả về từ model là tọa độ dạng % nên cần nhân với chiều rộng và chiều cao của ảnh
					cx, cy = int(lm.x * img_w), int(lm.y * img_h)

					# get the 2D/3D coordinates
					eye_2d.append([cx, cy])
					eye_3d.append([cx, cy, lm.z])

					# hiện index của tọa độ các điểm landmark cần biết của 
					# drawPoint(image, (cx, cy), label = str(idx))
				
			# convert to numpy array
			face_2d = np.array(face_2d, dtype=np.float64)
			face_3d = np.array(face_3d, dtype=np.float64)
			eye_2d = np.array(eye_2d, dtype=np.float64)
			eye_3d = np.array(eye_3d, dtype=np.float64)

			# the camera matrix
			# chỗ này chưa hiểu lắm
			focal_length = 1 * img_w

			cam_matrix = np.array([ [focal_length, 0, img_h/2],
									[0, focal_length, img_w/2],
									[0, 0, 1]]
									# , dtype=np.float64
								)
			
			# the distortion matrix
			dist_matrix = np.zeros((4,1), dtype=np.float64)

			# solve PnP
			# success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
			face_success, face_rotation_vector, face_translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
			eye_success, eye_rotation_vector, eye_translation_vector = cv2.solvePnP(eye_3d, eye_2d, cam_matrix, dist_matrix)

			# get rotation matrix
			face_rotation_matrix, f_jac = cv2.Rodrigues(face_rotation_vector)
			eye_rotation_matrix, f_jac = cv2.Rodrigues(eye_rotation_vector)

			# get angles
			face_angles, f_mtxR, f_mtxQ, f_Qx, f_Qy, f_Qz = cv2.RQDecomp3x3(face_rotation_matrix)
			# get the y rotation degree
			x = face_angles[0] * 360
			y = face_angles[1] * 360
			z = face_angles[2] * 360
			
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
			nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, face_rotation_vector, face_translation_vector, cam_matrix, dist_matrix)

			# p1 là tọa độ của mũi 
			p1 = (int(nose_2d[0]), int(nose_2d[1]))
			# p2 là tọa độ của mũi cộng với x và y
			# p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
			p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

			cv2.line(image, p1, p2, COLOR['BLUE'], 3)

			# Add the text on the image
			cv2.putText(image, txt(f'Face: {text}'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
			# cv2.putText(image, txt(f'Nose: x={str(np.round(x, 2)):4} y={str(np.round(y, 2)):4} z={str(np.round(z, 2)):4}'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, COLOR['RED'], 1)
			txt(f'Nose: x={str(np.round(x, 2)):4} y={str(np.round(y, 2)):4} z={str(np.round(z, 2)):4}')
			end = time.time()
			total_time = end - start

			fps = 1/total_time
			# cv2.putText(image, txt(f'FPS: {str(np.round(fps, 2))}'), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.25, COLOR['RED'], 1)
			txt(f'FPS: {str(np.round(fps, 2))}')

			# tính toán tỉ lệ mắt để xác định trạng thái mắt đang nhắm hay mở
			# thêm image vào để vẽ các đường line (if necessary)
			eyes_ratio = eyesDistances(landmarks, image)
			# cv2.putText(image, txt(f'eyes_ratio: {str(np.round(eyes_ratio, 2))}'), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.25, COLOR['RED'], 1)
			txt(f'eyes_ratio: {str(np.round(eyes_ratio, 2))}')
			
			if eyes_ratio > 5.3:
				cv2.putText(image, txt("Nham mat - Blink"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.25, COLOR['YELLOW'], 1)


			iris_position, iris_ratio, iris_color = irisDistance(landmarks, image)
			cv2.putText(image, txt(f'Iris: {iris_position}'), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.25, iris_color, 1)
			cv2.putText(image, txt(f'(ratio: {str(np.round(iris_ratio, 2))})'), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.25, COLOR['GREEN'], 1)
			
			txt('------------')


			# dùng tool có sẵn của lib để vẽ các connection giữa các điểm landmark
			# mp_drawing.draw_landmarks(
			# 	image=image,
			# 	landmark_list=face_landmarks,
			# 	connections=mp_face_mesh.FACEMESH_CONTOURS,
			# 	landmark_drawing_spec=drawing_spec,
			# 	connection_drawing_spec=drawing_spec
			# )


	
	# Flip the image horizontally for a selfie-view display.
	# cv2.imshow(screen_name, cv2.flip(image, 1))
	if (img_w < 800):
		image = cv2.resize(image, (round(img_w * 3), round(img_h * 3)))  
	cv2.imshow(screen_name, image)
	
# ============================================================= #

screen_name = '(MediaPipe) Head_pose & Eyes_tracker'
cv2.namedWindow(screen_name)        # Create a named window
cv2.moveWindow(screen_name, 60,30)  # Move it to (60,30)


PATHS = {
	'img_list': 'face_detection/raw/wider_face_split/wider_face_testdemo_filelist.txt',
	'img_dir': 'face_detection/raw/WIDER_test/images',
}
img_list_file = PATHS["img_list"]
# Read the list of image files
with open(img_list_file, 'r') as file:
	file_list = [line.strip() for line in file.readlines()]


with mp_face_mesh.FaceMesh(
	max_num_faces=1,
	refine_landmarks=True,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5) as face_mesh:

	for file_path in file_list:
		file_path = f'{PATHS["img_dir"]}/{file_path}'
		input_image = Image.open(file_path)
		
		# detect faces in image
		output_faces = model_service.predict_for(Constants.Models.FACE_MODEL, input_image)

		if (len(output_faces) > 0):
			for face_info in output_faces:
				face_img = face_info['face_img']    

				# convert from PIL image to numpy array to process by cv2
				face_img = np.array(face_img, dtype=np.uint8)
				detect_attention(face_img)
				
				key_press = cv2.waitKey(0) & 0xFF
				if key_press == ord('q'): 
					break
		