import cv2
import numpy as np
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# # For static images:
# IMAGE_FILES = []
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# with mp_face_mesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5) as face_mesh:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     # Convert the BGR image to RGB before processing.
#     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Print and draw face mesh landmarks on the image.
#     if not results.multi_face_landmarks:
#       continue
#     annotated_image = image.copy()
#     for face_landmarks in results.multi_face_landmarks:
#       print('face_landmarks:', face_landmarks)
#       mp_drawing.draw_landmarks(
#           image=annotated_image,
#           landmark_list=face_landmarks,
#           connections=mp_face_mesh.FACEMESH_TESSELATION,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp_drawing_styles
#           .get_default_face_mesh_tesselation_style())
#       mp_drawing.draw_landmarks(
#           image=annotated_image,
#           landmark_list=face_landmarks,
#           connections=mp_face_mesh.FACEMESH_CONTOURS,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp_drawing_styles
#           .get_default_face_mesh_contours_style())
#       mp_drawing.draw_landmarks(
#           image=annotated_image,
#           landmark_list=face_landmarks,
#           connections=mp_face_mesh.FACEMESH_IRISES,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp_drawing_styles
#           .get_default_face_mesh_iris_connections_style())
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ============================================================= #
def getIris(image, landmarks):
	right_iris_x = int(landmarks[473].x * image.shape[1])
	right_iris_y = int(landmarks[473].y * image.shape[0])
	left_iris_x = int(landmarks[468].x * image.shape[1])
	left_iris_y = int(landmarks[468].y * image.shape[0])
	
	return right_iris_x, right_iris_y, left_iris_x, left_iris_y

def getRightEye(image, landmarks):
	eye_top = int(landmarks[386].y * image.shape[0])
	eye_left = int(landmarks[362].x * image.shape[1])
	eye_bottom = int(landmarks[374].y * image.shape[0])
	eye_right = int(landmarks[263].x * image.shape[1])
	right_eye = image[eye_top:eye_bottom, eye_left:eye_right]
	return right_eye

def getRightEyeRect(image, landmarks):
	eye_top = int(landmarks[386].y * image.shape[0])
	eye_left = int(landmarks[362].x * image.shape[1])
	eye_bottom = int(landmarks[374].y * image.shape[0])
	eye_right = int(landmarks[263].x * image.shape[1])

	cloned_image = image.copy()
	cropped_right_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
	h, w, _ = cropped_right_eye.shape
	x = eye_left
	y = eye_top
	return x, y, w, h

def getLeftIris(image, landmarks):
	iris_y = int(landmarks[468].y * image.shape[0])
	iris_x = int(landmarks[468].x * image.shape[1])
	left_iris = image[iris_y:iris_y, iris_x:iris_x]
	return left_iris

def getLeftEye(image, landmarks):
	eye_top = int(landmarks[159].y * image.shape[0])
	eye_left = int(landmarks[33].x * image.shape[1])
	eye_bottom = int(landmarks[145].y * image.shape[0])
	eye_right = int(landmarks[133].x * image.shape[1])
	left_eye = image[eye_top:eye_bottom, eye_left:eye_right]
	return left_eye

def getLeftEyeRect(image, landmarks):
	# eye_left landmarks (27, 23, 130, 133) ->? how to utilize z info
	eye_top = int(landmarks[159].y * image.shape[0])
	eye_left = int(landmarks[33].x * image.shape[1])
	eye_bottom = int(landmarks[145].y * image.shape[0])
	eye_right = int(landmarks[133].x * image.shape[1])

	cloned_image = image.copy()
	cropped_left_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
	h, w, _ = cropped_left_eye.shape

	x = eye_left
	y = eye_top
	return x, y, w, h

def getLandmarks(image):
	face_mesh = mp_face_mesh.FaceMesh(
		min_detection_confidence=0.5, 
		min_tracking_confidence=0.5
	)
	# To improve performance, optionally mark the image as not writeable to
	# pass by reference.
	image.flags.writeable = False
	results = face_mesh.process(image)
	landmarks = results.multi_face_landmarks[0].landmark
	return landmarks, results


# ============================== Gaze direction ============================== #

def calculate_gaze_direction(landmarks):
	# Lấy vị trí của hai điểm mốc mắt (ví dụ: mắt trái và mắt phải)
	left_eye_x, left_eye_y = landmarks[36][0], landmarks[36][1]
	right_eye_x, right_eye_y = landmarks[39][0], landmarks[39][1]

	# Tính toán hướng nhìn ngang
	horizontal_gaze_direction = (right_eye_x - left_eye_x) / (right_eye_x + left_eye_x)

	# Tính toán hướng nhìn dọc
	vertical_gaze_direction = (right_eye_y - left_eye_y) / (right_eye_y + left_eye_y)

	# Trả về hướng nhìn dưới dạng tuple (hướng nhìn ngang, hướng nhìn dọc)
	return horizontal_gaze_direction, vertical_gaze_direction


def is_looking_at_camera(gaze_direction):
	# Xác định ngưỡng cho hướng nhìn ngang và dọc
	horizontal_threshold = 0.2
	vertical_threshold = 0.3

	# Kiểm tra xem hướng nhìn ngang có nằm trong ngưỡng hay không
	if abs(gaze_direction[0]) < horizontal_threshold:
		looking_at_camera_horizontally = True
	else:
		looking_at_camera_horizontally = False

	# Kiểm tra xem hướng nhìn dọc có nằm trong ngưỡng hay không
	if abs(gaze_direction[1]) < vertical_threshold:
		looking_at_camera_vertically = True
	else:
		looking_at_camera_vertically = False

	# Kiểm tra kết quả
	if looking_at_camera_horizontally and looking_at_camera_vertically:
		return True
	else:
		return False

# # ============================= way 2================================ #
# def calculate_gaze_direction(landmarks):
# 	"""
# 	Calculates gaze direction based on eye landmark positions.

# 	Args:
# 	landmarks: A list of facial landmark coordinates from MediaPipe's face_mesh solution.

# 	Returns:
# 	A tuple representing gaze direction (horizontal, vertical) in degrees.
# 	"""

# 	# Define eye landmark indices based on MediaPipe face mesh structure (you might need to adjust these based on your specific use case)
# 	left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
# 	right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# 	# Get eye landmarks
# 	left_eye = np.array([landmarks[i] for i in left_eye_indices])
# 	right_eye = np.array([landmarks[i] for i in right_eye_indices])

# 	# Calculate center of each eye
# 	left_eye_center = left_eye.mean(axis=0)
# 	right_eye_center = right_eye.mean(axis=0)

# 	print('-----left_eye_center, right_eye_center-----')
# 	print(left_eye_center, right_eye_center)

# 	# Assuming user is looking straight ahead when both eye centers have similar y-coordinates
# 	if abs(left_eye_center[1] - right_eye_center[1]) < 0.1:
# 	# User is looking straight ahead
# 		print('---->>> 0.0')
# 		return 0, 0
	
# 	print('---->>> gaze')

# 	# Calculate gaze direction vector (subtracting right eye center from left eye center)
# 	gaze_vector = left_eye_center - right_eye_center

# 	# Calculate horizontal and vertical gaze angles in degrees
# 	horizontal_gaze_angle = math.degrees(math.atan2(gaze_vector[0], abs(gaze_vector[2])))
# 	vertical_gaze_angle = math.degrees(math.atan2(gaze_vector[1], abs(gaze_vector[2])))

# 	return horizontal_gaze_angle, vertical_gaze_angle

# ============================= way 3 ================================ #
def calculate_gaze_direction(landmarks):
	"""
	Tính toán vectơ hướng nhìn dựa trên vị trí điểm mốc mắt.

	Args:
		landmarks: Danh sách tọa độ điểm mốc khuôn mặt từ giải pháp face_mesh của MediaPipe.

	Returns:
		Vectơ hướng nhìn 3D (x, y, z) biểu thị hướng nhìn.
	"""

	# Xác định chỉ số điểm mốc mắt dựa trên cấu trúc face_mesh của MediaPipe (có thể cần điều chỉnh dựa trên trường hợp sử dụng cụ thể)
	left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
	right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

	# Lấy điểm mốc mắt
	left_eye = np.array([landmarks[i] for i in left_eye_indices])
	right_eye = np.array([landmarks[i] for i in right_eye_indices])

	# Tính toán tâm mắt
	left_eye_center = left_eye.mean(axis=0)
	right_eye_center = right_eye.mean(axis=0)

	# Giả sử người dùng nhìn thẳng khi cả hai tâm mắt có tọa độ y tương tự
	if abs(left_eye_center[1] - right_eye_center[1]) < 0.1:
		# Người dùng nhìn thẳng
		return np.array([0, 0, 1])  # Giả sử hướng nhìn thẳng là dọc theo trục z

	# Tính toán vectơ hướng nhìn (trừ tâm mắt phải khỏi tâm mắt trái)
	gaze_vector = left_eye_center - right_eye_center

	# Chuẩn hóa vectơ hướng nhìn (làm cho nó có độ dài bằng 1)
	gaze_vector_norm = gaze_vector / np.linalg.norm(gaze_vector)

	# Tính toán góc nhìn ngang và dọc theo radian (điều chỉnh cho vectơ hướng nhìn 3D)
	horizontal_gaze_angle = math.atan2(gaze_vector_norm[0], gaze_vector_norm[2])
	vertical_gaze_angle = math.atan2(gaze_vector_norm[1], gaze_vector_norm[2])

	# Chuyển đổi góc nhìn thành tọa độ cầu (cho vectơ hướng nhìn 3D)
	x = math.sin(horizontal_gaze_angle) * math.cos(vertical_gaze_angle)
	y = math.sin(horizontal_gaze_angle) * math.sin(vertical_gaze_angle)
	z = math.cos(horizontal_gaze_angle)

	return np.array([x, y, z])



# Define font and text position for visualization
font = cv2.FONT_HERSHEY_SIMPLEX
text_position = (10, 30)

def draw_gaze_direction(image, horizontal_angle, vertical_angle):
	# Convert angles to integer values for text display
	horizontal_angle_int = int(horizontal_angle)
	vertical_angle_int = int(vertical_angle)

	# Prepare text strings
	horizontal_text = f"Horizontal Angle: {horizontal_angle_int}°"
	vertical_text = f"Vertical Angle: {vertical_angle_int}°"

	# Draw text on the image
	cv2.putText(image, horizontal_text, text_position, font, 1, (0, 255, 255), 1)
	cv2.putText(image, vertical_text, (text_position[0], text_position[1] + 20), font, 1, (0, 255, 255), 1)

# ============================================================= #

cap = cv2.VideoCapture(0)
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

		# To improve performance, optionally mark the image as not writeable to
		# pass by reference.
		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		results = face_mesh.process(image)
		

		# Convert the BGR image to RGB before processing.
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		

		# Print and draw face mesh landmarks on the image.
		if results.multi_face_landmarks:
			for face_landmarks in results.multi_face_landmarks:
				landmarks = face_landmarks.landmark
				
				# mp_drawing.draw_landmarks(
				# 	image=image,
				# 	landmark_list=face_landmarks,
				# 	connections=mp_face_mesh.FACEMESH_CONTOURS,
				# 	landmark_drawing_spec=drawing_spec,
				# 	connection_drawing_spec=drawing_spec)
				
				# mp_drawing.draw_landmarks(
				# 	image=image,
				# 	landmark_list=face_landmarks,
				# 	connections=mp_face_mesh.FACEMESH_LEFT_EYE,
				# 	landmark_drawing_spec=None,)

				# mp_drawing.draw_landmarks(
				# 	image=image,
				# 	landmark_list=face_landmarks,
				# 	connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
				# 	landmark_drawing_spec=None,)
				
				# mp_drawing.draw_landmarks(
				# 	image=image,
				# 	landmark_list=face_landmarks,
				# 	connections=mp_face_mesh.FACEMESH_IRISES,
				# 	landmark_drawing_spec=None,)
				
				# vẽ hình chữ nhật quanh mắt
				rightEyeImg = getRightEye(image, landmarks)
				rightEyeHeight, rightEyeWidth, _ = rightEyeImg.shape

				xRightEye, yRightEye, rightEyeWidth, rightEyeHeight = getRightEyeRect(image, landmarks)
				cv2.rectangle(image, (xRightEye, yRightEye),
							(xRightEye + rightEyeWidth, yRightEye + rightEyeHeight), (0, 0, 255), 2)

				# LEFT EYE
				leftEyeImg = getLeftEye(image, landmarks)
				leftEyeHeight, leftEyeWidth, _ = leftEyeImg.shape

				xLeftEye, yLeftEye, leftEyeWidth, leftEyeHeight = getLeftEyeRect(image, landmarks)
				cv2.rectangle(image, (xLeftEye, yLeftEye),
							(xLeftEye + leftEyeWidth, yLeftEye + leftEyeHeight), (0, 0, 255), 2)

				
				leftIrisX, leftIrisY, rightIrisX, rightIrisY = getIris(image, landmarks)
				cv2.circle(image, (leftIrisX, leftIrisY), radius=2, color=(0,255,255), thickness=2)
				cv2.circle(image, (rightIrisX, rightIrisY), radius=2, color=(0,255,255), thickness=2)


				# # Extract landmarks
				# landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark])
				# print(landmarks)
				# # Calculate gaze direction
				# horizontal_angle, vertical_angle = calculate_gaze_direction(landmarks)
				# print(horizontal_angle, vertical_angle)
				

				# if (horizontal_angle != 0 and vertical_angle != 0):
				# 	break

				# # Draw gaze direction text on the frame
				# draw_gaze_direction(image, horizontal_angle, vertical_angle)
						
				# Draw other desired information or visualizations based on landmarks (optional)
				# mp_drawing.draw_landmarks(
				# 	image=image,
				# 	landmark_list=face_landmarks,
				# 	connections=mp_face_mesh.FACEMESH_TESSELATION,
				# 	landmark_drawing_spec=None,
				# 	connection_drawing_spec=mp_drawing_styles
				# 	.get_default_face_mesh_tesselation_style())

		
		# Flip the image horizontally for a selfie-view display.
		cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
		cv2.imshow('MediaPipe Face Mesh', image)
		if cv2.waitKey(0) & 0xFF == 27:
			break
cap.release()