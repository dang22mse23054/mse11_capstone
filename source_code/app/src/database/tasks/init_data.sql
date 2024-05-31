INSERT INTO capstone.video (id,title,refFileName,refFilePath,isEnabled,deletedAt,createdAt,updatedAt) VALUES
	 (1,'Quảng cáo ô tô','Young_Car.mp4','video/1/20240331_0450_eyq10mvfg7.mp4',1,NULL,'2024-03-31 04:50:57','2024-03-31 04:50:57'),
	 (2,'Dụng cụ bếp','Woman_Home.mp4','video/2/20240331_0451_v4wwcx7hy.mp4',1,NULL,'2024-03-31 04:52:41','2024-03-31 04:52:41'),
	 (3,'Váy nữ','Woman_Dress.mp4','video/3/20240331_0455_agkjte28k3.mp4',1,NULL,'2024-03-31 04:56:39','2024-03-31 04:56:39'),
	 (4,'Nước hoa nam','Men_Perfume.mp4','video/4/20240331_0458_y6mcdhj0k.mp4',1,NULL,'2024-03-31 04:58:46','2024-03-31 04:58:46'),
	 (5,'Trường quốc tế cho thiếu niên','Kid_School.mp4','video/5/20240331_0459_w2n2hx7j.mp4',1,NULL,'2024-03-31 05:00:08','2024-03-31 05:00:08'),
	 (6,'Trại hè cho trẻ','Kid_Campaign.mp4','video/6/20240331_0500_9z5qzovnd.mp4',1,NULL,'2024-03-31 05:00:38','2024-03-31 05:00:38'),
	 (7,'Du lịch nghỉ dưỡng đảo','Everyone_Travel.mp4','video/7/20240331_0500_dtdoxukt3.mp4',1,NULL,'2024-03-31 05:01:35','2024-03-31 05:01:35'),
	 (8,'Thức ăn nhanh ','Everyone_Food.mp4','video/8/20240331_0501_qih0rswu2m.mp4',1,NULL,'2024-03-31 05:02:05','2024-03-31 05:02:05'),
	 (9,'Khu du lịch nghỉ dưỡng cho người lớn tuổi ','Elderly_care.mp4','video/9/20240331_0502_o9uae8vcfi.mp4',1,NULL,'2024-03-31 05:02:57','2024-03-31 05:02:57');


INSERT INTO capstone.video_category (id,videoId,categoryId) VALUES 
	(1,1,5), (2,1,6), (3,1,7), (4,2,4), (5,2,6), (6,3,4), (7,3,6), (8,4,3), (9,4,5), (10,4,7), (11,5,3), (12,5,4), (13,6,1), 
	(14,6,2), (15,6,3), (16,6,4), (17,7,5), (18,7,6), (19,7,7), (20,7,8), (21,7,9), (22,7,10), (23,8,1), (24,8,2), (25,8,3), 
	(26,8,4), (27,8,5), (28,8,6), (31,9,7), (32,9,8), (29,9,9), (30,9,10);


