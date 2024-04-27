import os, random, uuid, time
import cv2, requests, numpy as np
from multiprocessing import Process
from threading import Thread
from services.cache_service import CacheService
from common.constants import Constants
from utils.decoder import txt_color
from PIL import Image
import traceback

import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)


VIDEO_FOLDER = 'video'
UNKNOWN = 'Unknown'
DEFAULT_VIDEO = 'default_video.mp4'
all_ads = os.listdir(VIDEO_FOLDER)
all_ads.remove(DEFAULT_VIDEO)
DEFAULT_VIDEO = f'{VIDEO_FOLDER}/{DEFAULT_VIDEO}'

AGE = Constants.Age()
AGES = list(AGE.Groups.keys())

status_path = f'{VIDEO_FOLDER}/status.cache'
status_cache_service = CacheService(status_path)

# format: videoId|video_name
advice_ads_cache_path = f'{VIDEO_FOLDER}/advice_ads.cache'
advice_ads_cache_service = CacheService(advice_ads_cache_path)

all_valid_ads_cache_path = f'{VIDEO_FOLDER}/all_valid_ads.cache'
all_valid_ads_cache_service = CacheService(all_valid_ads_cache_path, 'list')

def reset_caches():
	advice_ads_cache_service.delete()
	all_valid_ads_cache_service.delete()
	status_cache_service.set('Run')
 
def is_running_server():
    return status_cache_service.get() == 'Run'

def stop_server():
    status_cache_service.set('Stop')

def show_bbox(image, faces, screen_name, screen_ratio = 5):
	# image = image.copy()

	for index, face in enumerate(faces):
		score = face['score']
		bbox = face['bbox']
		bbox = np.array(bbox, dtype=np.int32)
		label = f"{'Nam' if face['gender'] == 0 else 'Nu'}-{AGES[face['age']]}"
		cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
  
		cv2.putText(
			image,
			label,
			(int(bbox[0]), int(bbox[1]) - 10),
			fontFace = cv2.FONT_HERSHEY_DUPLEX,
			fontScale = 2,
			color = (0,0,255),
			thickness=3,
			lineType=cv2.LINE_AA
		)

	height, width = image.shape[:2]
	image = cv2.resize(image, (round(width / screen_ratio), round(height / screen_ratio)))  
	cv2.imshow(screen_name, image)

# ============= Sync all valid ads from server, interval 5s ============= #
def sync_all_valid_ads_cache():
	api_headers = {
		# 'Content-Type': 'multipart/form-data',
	}
	while is_running_server():
		try:
			response = requests.get(
				'https://mse11-capstone.com.vn/dapi/ads/all', 
				headers=api_headers,
				# reject self-signed SSL certificates
				verify=False
			)
			data = response.json()['data']
		
			if (len(data) > 0):
				cache_data = []
				for video in data:
					v_name = list(video.keys())[0]
					cache_data.append(f'{video[v_name]}|{v_name}')

				all_valid_ads_cache_service.set(cache_data)
			
		except Exception as e:
			print(traceback.format_exc())

		time.sleep(5)

# ========================== Ads Player ========================== #
def trigger_ads_player():
    # Player settings
	screen_name = 'Advertisement'
	cv2.namedWindow(screen_name)        # Create a named window
	cv2.moveWindow(screen_name, 600,30)  # Move it to (40,30)
	screen_ratio = 2
 
	# Player process
	video_player = None
 
	time.sleep(3)
	while is_running_server():
		print('====== Play ads ======')
		# get all valid ads from cache
		sync_all_ads = all_valid_ads_cache_service.get()
		print(f'> Loading all Ads (Cached)\n {sync_all_ads}')
	
		# If there is no valid ads, play all ads at local folder
		all_ads = sync_all_ads if len(sync_all_ads) > 0 else all_ads
		print(f'> Available ads:\n {all_ads}')
	
		# Get adviced ads from cache (if any)
		# format: videoId|video_name
		adviced_ads_id = 0
		adviced_ads_name = None
		adviced_ads = advice_ads_cache_service.get()
  
		if (adviced_ads is not None or len(all_ads) > 0):
			adviced_ads = (random.choice(all_ads) if adviced_ads is None else adviced_ads).split('|')
			adviced_ads_id = adviced_ads[0]
			adviced_ads_name = adviced_ads[1]
  
		# Select adviced ads. If there is no adviced ads left, select a random video from all ads. If no ads left, play default video
		ads_path = f"{VIDEO_FOLDER}/{adviced_ads_name}" if adviced_ads_name is not None else DEFAULT_VIDEO
	
		print(f'> ads_path = {ads_path}')
   
		print(f'> Playing {ads_path}')
   
		# Create a VideoCapture object and read from input file 
		video_player = cv2.VideoCapture(ads_path)
	
		# Check if ads is opened successfully 
		if (video_player.isOpened()== False): 
			print(f"Error opening video file {ads_path} -> play default video") 
			video_player = cv2.VideoCapture(DEFAULT_VIDEO) 
	
		# Calc duration of video
		fps = video_player.get(cv2.CAP_PROP_FPS)
		frame_count = int(video_player.get(cv2.CAP_PROP_FRAME_COUNT))
  
		# Read until video is completed
		removed_cache = False
		while(video_player.isOpened()): 
			
			# Capture frame-by-frame 
			ret, frame = video_player.read() 
			remain_frame_num = frame_count - int(video_player.get(cv2.CAP_PROP_POS_FRAMES))
			remain_seconds = int((remain_frame_num/fps)%60)
   
			# When video has remaining 8s to be done, release cache
			if adviced_ads is not None and remain_seconds < 8 and removed_cache == False:
				advice_ads_cache_service.delete()
				removed_cache = True
    
			if ret == True: 
				# Display the resulting frame 
				height, width = frame.shape[:2]
				frame = cv2.resize(frame, (round(width / screen_ratio), round(height / screen_ratio)))  
				cv2.imshow(screen_name, frame) 
			
			# Break the loop 
			else: 
				break
		
			# Press Q on keyboard to exit 
			key_press = cv2.waitKey(25) & 0xFF
			if key_press == ord('q'): 
				stop_server()
				break

			# Press N on keyboard to skip to next video
			elif key_press == ord('n'): 
				break

			# Press S on keyboard to show adviced ads
			elif key_press == ord('s'):
				print(f"[{screen_name}] Adviced ads")
				print(advice_ads_cache_service.get())

			# Press A on keyboard to add to adviced ads
			elif key_press == ord('a'):
				print(f"[{screen_name}] Add to Adviced ads")
				advice_ads_cache_service.set('video2.mp4')
   
		time.sleep(2)
	
	if (video_player is not None):
		# When everything done, release 
		# the video capture object 
		video_player.release() 
	
	# Closes all the frames 
	cv2.destroyAllWindows() 
    
# ========================== Camera Tracker ========================== #
def trigger_cam_tracker(using_image = False):
	screen_name = 'CamTracker'
	cam_tracker = None
	recently_ads_list = []
	cv2.namedWindow(screen_name)        # Create a named window
	cv2.moveWindow(screen_name, 100,30)  # Move it to (40,30)
 
	log_screen_name = 'AdsLog'
	cv2.namedWindow(log_screen_name)        # Create a named window
	cv2.moveWindow(log_screen_name, 600,300)  # Move it to (40,30)
 
	api_headers = {
		# 'Content-Type': 'multipart/form-data',
	}

	if (using_image):
		# Dùng TEST folder (TEST_IMAGE) ===== BEGIN
		PATHS = {
			'img_list': 'face_detection/raw/wider_face_split/wider_face_testdemo_filelist.txt',
			'img_dir': 'face_detection/raw/WIDER_test/images',
		}
		img_list_file = PATHS["img_list"]
		# Read the list of image files
		with open(img_list_file, 'r') as file:
			file_list = [line.strip() for line in file.readlines()]
		
	else:
		cam_tracker = cv2.VideoCapture(0)

	while is_running_server():
		try:
			# Prepare frame to image for sending to server
			frame = None
			if (using_image):
				# Randomly select an image
				file_path = f'{PATHS["img_dir"]}/{random.choice(file_list)}'
				ori_img = Image.open(file_path)
				# Chuyển đổi ảnh từ PIL sang OpenCV
				frame = np.array(ori_img)
			else:
				is_success, frame = cam_tracker.read()
				if not is_success:
					continue

			# Chuyển đổi khung hình thành định dạng chuỗi để gửi qua HTTP
			_, img_encoded = cv2.imencode('.jpg', frame)
			frame_as_bytes = img_encoded.tobytes()
	
			# height, width = frame.shape[:2]
			# frame = cv2.resize(frame, (round(width / screen_ratio), round(height / screen_ratio)))  
			# cv2.imshow(screen_name, frame)
   
			# Tạo dữ liệu để gửi
			# phải set tên file, file type thì NodeJS server mới xác định được
			file_obj = {'uploadedFile': (f"{uuid.uuid4()}.jpg", frame_as_bytes, 'image/jpeg')}
   
			# Default variable value
			adviced_ads_id = 0
			adviced_ads_name = None

			# only add cache if there is no cache
			adviced_ads = advice_ads_cache_service.get()

			# FOR DEBUG
			# adviced_ads = '12|daka.mp4'
			# If not have any adviced ads, call API to get adviced ads
			if (adviced_ads is None):
				print(txt_color('**************** Get ADVICE ads ****************', Constants.TXT_COLOR['GREEN']))
				
				try:
					response = requests.post(
						'https://mse11-capstone.com.vn/dapi/ads/advice', 
						headers=api_headers,
						files = file_obj,
						# reject self-signed SSL certificates
						verify=False
					)

					# Lấy thông tin từ response
					data = response.json()['data']

					suggested_videos = data.get('videos', None)
					print(' > === Suggested videos ===')
					print(suggested_videos)
					if (suggested_videos is not None and len(suggested_videos) > 0):
						majority_age = data.get('majority_age', None)
						majority_gender = data.get('majority_gender', None)

						logTxt = ' > [Suggestion] === Response data ===\n'
						logTxt += f"  majority_age={AGES[majority_age] if majority_age else UNKNOWN}\n"
						logTxt += f"  majority_gender={('Female' if majority_gender else 'Male') if majority_gender is not None else UNKNOWN}\n"
						logTxt += f"  suggested_videos={suggested_videos}"
						print(logTxt)

						# remove ads the same with the last adviced video
						# calc recently ads
						if (len(recently_ads_list) > 0):
							# if there is any recently ads, remove it from suggested videos
							for recently_ads in recently_ads_list:
								if recently_ads in suggested_videos:
									suggested_videos.remove(recently_ads)
							print(f'* recently_ads_list = {recently_ads_list}')
							print(f'* suggested_videos = {suggested_videos}')

						s_video = random.choice(suggested_videos)
						print(f' > Set suggested video: {s_video}')

						# get video name and video id (using for loop because s_video is the dict, not the list)
						for video_name in s_video:
							advice_ads_cache_service.set(f'{s_video[video_name]}|{video_name}')
							
							# add suggested video to recently ads list
							recently_ads_list.append(s_video)
							if (len(recently_ads_list) > 3):
								# only keep max 3 items in recently ads list
								recently_ads_list = recently_ads_list[1:]

					faces = data['faces']
					if (len(faces) > 0):
						show_bbox(frame, faces, screen_name)

				except requests.exceptions.RequestException as e:
					print(traceback.format_exc())
				
				# cv2.waitKey(2) trả về một giá trị int, nếu giá trị trả về khác 0 thì sẽ chờ x giây, tiếp tục chạy vòng lặp, ngược lại sẽ thoát khỏi vòng lặp
				# 0xFF nghĩa là 11111111 trong hệ cơ số 2 tức là nó sẽ lấy 8 bit cuối cùng của kết quả trả về từ hàm cv2.waitKey(0) sau đó so sánh với giá trị của phím 'q' trong bảng mã ASCII
				# ord('q') trả về giá trị của phím 'q' trong bảng mã ASCII, có kiểu dữ liệu là int
				
				# Đợi 3 giây trước khi chụp ảnh tiếp theo
				# Press Q on keyboard to exit 
				key_press = cv2.waitKey(1000) & 0xFF
				if key_press == ord('q'): 
					stop_server()
					break
		
			else:
				# Logging the ads space to server for analytic later
				adviced_ads_id = adviced_ads.split('|')[0]
				print(txt_color(f'**************** Log ads ({adviced_ads_id}) to server ****************', Constants.TXT_COLOR['BLUE']))
				show_bbox(frame, [], log_screen_name)

				response = requests.post(
					f'https://mse11-capstone.com.vn/dapi/ads/log/{adviced_ads_id}', 
					headers=api_headers,
					files = file_obj,
					# reject self-signed SSL certificates
					verify=False
				)

				# Lấy thông tin từ response
				data = response.json()

				print(f' > [Logging] === Response data ===')
				print(data)
				# ko dc dùng time.sleep vì sẽ ko hiện dc cv2 window để show image
				# time.sleep(5)
				# Press Q on keyboard to exit 
				key_press = cv2.waitKey(5000) & 0xFF
				if key_press == ord('q'): 
					stop_server()
					break
	
		except Exception as e:
			print(traceback.format_exc())

	if (cam_tracker):
		cam_tracker.release()
	cv2.destroyAllWindows()

# ========================== MAIN ========================== #
if __name__ == "__main__":

	# Reset old cache
	reset_caches()
 
	print('[MAIN] Running...')
	procs = list()
	
	sync_all_valid_ads_proc = Process(target=sync_all_valid_ads_cache, args=())
	ads_player_proc = Process(target=trigger_ads_player, args=())
	cam_tracker_proc = Process(target=trigger_cam_tracker, args=(True,))
	
	procs.append(sync_all_valid_ads_proc)
	procs.append(ads_player_proc)
	procs.append(cam_tracker_proc)
	
	# Start processes
	for index, proc in enumerate(procs):
		print('[MAIN] Starting process ', index)
		proc.start()

		# DON'T USE join() HERE => Reason: next process cannot start until the previous one finished
		# join to guarantee Main will be closed after all threads finished
		# proc.join()
	
	# Listen to processes
	for index, proc in enumerate(procs):
		# join to guarantee Main will be closed after all threads finished
		proc.join()


	# Clear cache
	reset_caches()
	print('[MAIN] Stopped')
    