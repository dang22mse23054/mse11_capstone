import cv2, sys
import numpy as np
import mediapipe as mp
sys.path.append('../')

# Initialize MediaPipe Face and Pose modules
mp_face = mp.solutions.face_detection

# Initialize Face Detection and Pose Estimation models
face_detection = mp_face.FaceDetection(
	model_selection=1,
	min_detection_confidence=0.3
)

from face_detection.dataset import PATHS
from common.constants import Constants
screen_ratio = 2

MODE = Constants.Mode()
img_list_file = f'../face_detection/{PATHS[MODE.TESTDEMO]["img_list"]}'
# Read the list of image files
with open(img_list_file, 'r') as file:
	file_list = [line.strip() for line in file.readlines()]

for file_path in file_list:
	file_path = f'../face_detection/{PATHS[MODE.TESTDEMO]["img_dir"]}/{file_path}'
	img = cv2.imread(file_path)

	# Convert the frame to RGB format for MediaPipe
	frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	ih, iw, _ = frame_rgb.shape

	# Perform face detection
	face_results = face_detection.process(frame_rgb)
	print('=================')
	detected_num = len(face_results.detections) if face_results.detections else 0
	print(detected_num)

	if face_results.detections:
		for detection in face_results.detections:
			print(detection)
			
			bboxC = detection.location_data.relative_bounding_box
			
			x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
			
			cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
			# Perform pose estimation using the face region
			face_center = (x + w // 2, y + h // 2)
			frame_pose = frame_rgb[y:y + h, x:x + w]
			frame_pose_rgb = cv2.cvtColor(frame_pose, cv2.COLOR_BGR2RGB)

	screen_name = f"Detect: {detected_num} faces"
	cv2.namedWindow(screen_name)        # Create a named window
	cv2.moveWindow(screen_name, 100,30)  # Move it to (40,30)

	if (iw > 800):
		frame_rgb = cv2.resize(frame_rgb, (round(iw / screen_ratio), round(ih / screen_ratio)))  
	cv2.imshow(screen_name, frame_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()