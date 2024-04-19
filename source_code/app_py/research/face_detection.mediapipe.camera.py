import cv2
import mediapipe as mp

# Initialize MediaPipe Face and Pose modules
mp_face = mp.solutions.face_detection

# Initialize Face Detection and Pose Estimation models
face_detection = mp_face.FaceDetection(
	model_selection=1,
	min_detection_confidence=0.95
)

cap = cv2.VideoCapture(0)  # Use the desired camera index (e.g., 0 for the default camera)
cv2.namedWindow('Detection')        # Create a named window
cv2.moveWindow('Detection', 100,30)  # Move it to (40,30)

while True:
	success, frame = cap.read()
	if not success:
		continue

	# Convert the frame to RGB format for MediaPipe
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Perform face detection
	face_results = face_detection.process(frame_rgb)
	print('=================')
	print(len(face_results.detections) if face_results.detections else 0)

	if face_results.detections:
		for detection in face_results.detections:
			
			bboxC = detection.location_data.relative_bounding_box
			ih, iw, _ = frame.shape
			x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			# Perform pose estimation using the face region
			face_center = (x + w // 2, y + h // 2)

			# frame chưa khuôn mặt thôi
			frame_pose = frame[y:y + h, x:x + w]

		   
	cv2.imshow("Detection", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()