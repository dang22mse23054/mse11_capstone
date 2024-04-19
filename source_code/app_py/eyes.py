import cv2
import numpy as np
import mediapipe as mp
import time, math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ============================================================= #

def drawPoint(image, point, label = 'unknown'):
	cv2.putText(image, str(label), point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

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
	if ratio < 0.52:
		position = "RIGHT"
	elif 0.52 <= ratio and ratio <= 0.59:
		position = "CENTER" 
	else:
		position = "LEFT" 
	
	# logging
	info = f"Iris: center_to_right={np.round(center_to_right, 2):4}, center_to_left={np.round(center_to_left, 2):4}, total={np.round(total, 2):4}, ratio={np.round(ratio, 2):4} => {position}"
	if image is not None:
		cv2.putText(image, info, (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	else:
		print(info)

	return position, ratio

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
	if image is not None:
		img_h, img_w, img_c = image.shape

		# vẽ các đường giữa top-bottom, left-right của mắt 
  		# -- vẽ đường ngang (left-right)
		eye_pLeft = (int(landmarks[362].x * img_w), int(landmarks[362].y * img_h))
		drawPoint(image, eye_pLeft, label = '362')
		eye_pRight = (int(landmarks[263].x * img_w), int(landmarks[263].y * img_h))
		drawPoint(image, eye_pRight, label = '263')
		
		cv2.line(image, eye_pLeft, eye_pRight, (255, 255,0), 2)
		# hiện index của tọa độ các điểm landmark cần biết của 

		# -- vẽ đường dọc (top-bottom)
		eye_pTop = (int(landmarks[386].x * img_w), int(landmarks[386].y * img_h))
		drawPoint(image, eye_pTop, label = '386')
		eye_pBottom = (int(landmarks[374].x * img_w), int(landmarks[374].y * img_h))
		drawPoint(image, eye_pBottom, label = '374')
		
		cv2.line(image, eye_pTop, eye_pBottom, (255, 255,0), 2)

		# print('----------')
		# print(lh_right.x, lh_right.y)
		# print(lh_left.x, lh_left.y)
		# print(lv_top.x, lv_top.y)
		# print(lv_bottom.x, lv_bottom.y)
		# rv_top = int(rv_top.y * img_w)
		# rh_left = int(rh_left.x * img_h)
		# rv_bottom = int(rv_bottom.y * img_w)
		# rh_right = int(rh_right.x * img_h)
		
		# cv2.line(img, lh_right, lh_left, (255, 0,0), 2)
		# cv2.line(img, (lv_top.x, lv_top.y), (lv_bottom.x, lv_bottom.y), (0, 0,0), 2)
		# cv2.line(img, (rh_right.x, rh_right.y), (rh_left.x, rh_left.y), (255, 0,0), 2)
		# cv2.line(img, (rv_top.x, rv_top.y), (rv_bottom.x, rv_bottom.y), (0, 0,0), 2)

	return ratio

# ============================================================= #

cap = cv2.VideoCapture(0)
screen_name = '(MediaPipe) Head_pose & Eyes_tracker'
cv2.namedWindow(screen_name)        # Create a named window
cv2.moveWindow(screen_name, 60,30)  # Move it to (60,30)

with mp_face_mesh.FaceMesh(
	max_num_faces=1,
	refine_landmarks=True,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5) as face_mesh:
  
	while cap.isOpened():
		success, image = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			# If loading a video, use 'break' instead of 'continue'.
			continue

		start = time.time()

		# To improve performance, optionally mark the image as not writeable to
		# pass by reference.
		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# https://www.youtube.com/watch?v=-toNMaS4SeQ
		img_h, img_w, img_c = image.shape
		face_3d = []
		face_2d = []

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
						drawPoint(image, (cx, cy), label = str(idx))
				
				# convert to numpy array
				face_2d = np.array(face_2d, dtype=np.float64)
				face_3d = np.array(face_3d, dtype=np.float64)

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
				success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

				# get rotation matrix
				rotation_matrix, jac = cv2.Rodrigues(rotation_vector)

				# get angles
				angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)
				# get the y rotation degree
				x = angles[0] * 360
				y = angles[1] * 360
				z = angles[2] * 360

				if y < -3.7:
					text = "RIGHT"
				elif y > 3.7:
					text = "LEFT"
				elif x < -1.2:
					text = "DOWN"
				elif x > 4:
					text = "UP"
				else:
					text = "Forward"


				# display the nose direction
				nose_3d_projected, jacobian = cv2.projectPoints(nose_3d, rotation_vector, translation_vector, cam_matrix, dist_matrix)

				p1 = (int(nose_2d[0]), int(nose_2d[1]))
				p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

				cv2.line(image, p1, p2, (255, 0, 0), 3)

				# Add the text on the image
				cv2.putText(image, f'Face: {text}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
				cv2.putText(image, f'Nose: x={str(np.round(x, 2)):4} y={str(np.round(y, 2)):4} z={str(np.round(z, 2)):4}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

				end = time.time()
				total_time = end - start

				fps = 1/total_time
				cv2.putText(image, f'FPS: {str(np.round(fps, 2))}', (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

				# tính toán tỉ lệ mắt để xác định trạng thái mắt đang nhắm hay mở
				# thêm image vào để vẽ các đường line (if necessary)
				eyes_ratio = eyesDistances(landmarks, image)
				cv2.putText(image, f'eyes_ratio: {str(np.round(eyes_ratio, 2))}', (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

				if eyes_ratio > 5.3:
					cv2.putText(image, "Nham mat - Blink", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


				iris_position, iris_ratio = irisDistance(landmarks, image)
				cv2.putText(image, f'Iris: {iris_position}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
				cv2.putText(image, f'(ratio: {str(np.round(iris_ratio, 2))})', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
				


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
		cv2.imshow(screen_name, image)
		if cv2.waitKey(5) & 0xFF == 27:
			break
	cap.release()