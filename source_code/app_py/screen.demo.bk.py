import os, random, uuid, time, threading
import cv2, requests

VIDEO_FOLDER = 'video'
DEFAULT_VIDEO = 'default_video.mp4'
all_ads = os.listdir(VIDEO_FOLDER)
all_ads.remove(DEFAULT_VIDEO)
DEFAULT_VIDEO = f'{VIDEO_FOLDER}/{DEFAULT_VIDEO}'

adviced_ads = []

def play_ads_player():
    # Player settings
	screen_name = 'Advertisement'
	cv2.namedWindow(screen_name)        # Create a named window
	cv2.moveWindow(screen_name, 40,30)  # Move it to (40,30)
 
	# Player process
	video_player = None
	stop = False
 
	while not stop:
		# Select adviced ads. If there is no adviced ads left, select a random video from all ads. If no ads left, play default video
		ads_path = f"{VIDEO_FOLDER}/{adviced_ads.pop(0)}"  if len(adviced_ads) > 0 \
			else f"{VIDEO_FOLDER}/{random.choice(all_ads)}" if len(all_ads) > 0 \
			else DEFAULT_VIDEO
   
		# # Create a VideoCapture object and read from input file 
		video_player = cv2.VideoCapture(ads_path) 
	
		# Check if ads is opened successfully 
		if (video_player.isOpened()== False): 
			print(f"Error opening video file {ads_path} -> play default video") 
			video_player = cv2.VideoCapture(DEFAULT_VIDEO) 
	
		# Read until video is completed 
		while(video_player.isOpened()): 
			
			# Capture frame-by-frame 
				ret, frame = video_player.read() 
				if ret == True: 
					# Display the resulting frame 
					height, width = frame.shape[:2]
					frame = cv2.resize(frame, (round(width / 3), round(height / 3)))  
					cv2.imshow(screen_name, frame) 
					
				# Press Q on keyboard to exit 
					key_press = cv2.waitKey(25) & 0xFF
					if key_press == ord('q'): 
						stop = True
						break
  
					elif key_press == ord('n'): 
						break
			
			# Break the loop 
				else: 
					break
		
		time.sleep(2)
	
	if (video_player is not None):
		# When everything done, release 
		# the video capture object 
		video_player.release() 
	
	# Closes all the frames 
	cv2.destroyAllWindows() 
    
def trigger_cam_tracker():
	screen_name = 'Webcam'
	cv2.namedWindow(screen_name)        # Create a named window
	cv2.moveWindow(screen_name, 40,30)  # Move it to (40,30)
 
	api_url = 'https://mse11-capstone.com.vn/dapi/ads/advice'
	api_headers = {
		# 'Content-Type': 'multipart/form-data',
	}
 
	cam_tracker = cv2.VideoCapture(0)
	is_success, frame = cam_tracker.read()

	while True:
		is_success, frame = cam_tracker.read()
		if not is_success:
			break

		# Chuyển đổi khung hình thành định dạng chuỗi để gửi qua HTTP
		_, img_encoded = cv2.imencode('.jpg', frame)
		frame_as_bytes = img_encoded.tobytes()
  
		height, width = frame.shape[:2]
		frame = cv2.resize(frame, (round(width / 5), round(height / 5)))  
		cv2.imshow(screen_name, frame)
  
		# cv2.waitKey(2) trả về một giá trị int, nếu giá trị trả về khác 0 thì sẽ chờ x giây, tiếp tục chạy vòng lặp, ngược lại sẽ thoát khỏi vòng lặp
		# 0xFF nghĩa là 11111111 trong hệ cơ số 2 tức là nó sẽ lấy 8 bit cuối cùng của kết quả trả về từ hàm cv2.waitKey(0) sau đó so sánh với giá trị của phím 'q' trong bảng mã ASCII
		# ord('q') trả về giá trị của phím 'q' trong bảng mã ASCII, có kiểu dữ liệu là int
		
		# Tạo dữ liệu để gửi
		try:
			# phải set tên file, file type thì NodeJS server mới xác định được
			file_obj = {'uploadedFile': (f"{uuid.uuid4()}.jpg", frame_as_bytes, 'image/jpeg')}
			print("[CamTracker] Sending img...")
   
			# response = requests.post(
			# 	api_url, 
			# 	headers=api_headers,
			# 	files = file_obj,
			# 	# reject self-signed SSL certificates
    		# 	verify=False
			# )
			# print("Response:", response.text)
		except requests.exceptions.RequestException as e:
			print("Error:", e)
			break
		
  
  		# Đợi 3 giây trước khi chụp ảnh tiếp theo
		if cv2.waitKey(3000) & 0xFF == ord('q'):
			break

		# time.sleep(2)
		
	cam_tracker.release()
	cv2.destroyAllWindows()
 

if __name__ == "__main__":
	# cam_tracker1 = cv2.VideoCapture(0)
	# cam_tracker2 = cv2.VideoCapture(0)
	
	# while True:
	# 	is_success1, frame1 = cam_tracker1.read()
	
	# 	is_success2, frame2 = cam_tracker2.read()
  
	# 	# Chuyển đổi khung hình thành định dạng chuỗi để gửi qua HTTP
  
	# 	height, width = frame1.shape[:2]
	# 	frame1 = cv2.resize(frame1, (round(width / 5), round(height / 5)))  
  
	# 	height, width = frame2.shape[:2]
	# 	frame2 = cv2.resize(frame2, (round(width / 5), round(height / 5)))  
		
	# 	cv2.imshow('screen_name_1', frame1)
	# 	cv2.imshow('screen_name_2', frame2)
  
	# 	# Đợi 3 giây trước khi chụp ảnh tiếp theo
	# 	if cv2.waitKey(2000) & 0xFF == ord('q'):
	# 		break

	# 	# time.sleep(2)
		
	# cam_tracker1.release()
	# cam_tracker2.release()
 
	# cv2.destroyAllWindows()
    
	# threads = list()
	
	# print('[MAIN] Running...')
	
	# ads_player_thread = threading.Thread(target=play_ads_player, args=())
	# trigger_cam_tracker = threading.Thread(target=trigger_cam_tracker, args=())
	
	# threads.append(ads_player_thread)
	# threads.append(trigger_cam_tracker)
	
	# # # Start threads
	# for index, thread in enumerate(threads):
	# 	thread.start()
	# 	# join to guarantee Main will be closed after all threads finished
	# 	thread.join()

	# print('[MAIN] Stopped')
    