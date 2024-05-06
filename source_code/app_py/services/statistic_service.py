import sys
sys.path.append('../')

import os, json
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timezone
from models.db.log import Log
from models.db.statistic import Statistic
from models.db._base import DbBase
from utils.timer import TimeUtils
from utils.decoder import NumpyEncoder, txt_color

from sqlalchemy import and_, or_, not_, tuple_, text, select, func, case, literal_column, delete
from sqlalchemy.orm import aliased

TIME_ZONE = "UTC"
l = aliased(Log, name='l')
s = aliased(Statistic, name='s')

class StatisticService():

	def calc_hourly_statistic(self, cuz_time = None):
		session = None

		try:
			time_obj = datetime.fromisoformat(cuz_time) if cuz_time else TimeUtils.now()
			
			from_time = TimeUtils.get_start_of_hours(time_obj, -1, TIME_ZONE)
			print(from_time)
			to_time = TimeUtils.get_start_of_hours(time_obj, 0, TIME_ZONE)
			print(to_time)

			statistic_group = from_time.strftime("%Y%m%d%H")

			# get all video ids that have been processed in the last hour
			
			# Cách 1: Sử dụng custom query
			stm = select([l.videoId.distinct().label("videoId")]) \
				.select_from(l) \
				.where(and_(l.createdAt >= from_time, l.createdAt < to_time))
			
			log_list = DbBase.exec_custom_query(stm, True)
			video_ids = [log.videoId for log in log_list]

			# Cách 2: Sử dụng ORM
			# session = DbBase.get_session(True)
			# try:
			# 	log_list = session.query(l.videoId.distinct().label("videoId")) \
			# 		.filter(and_(l.createdAt >= from_time, l.createdAt < to_time)) \
			# 		.all()
				
			# 	videoIds = [log.videoId for log in log_list]
			# 	print(videoIds)
			# except Exception as e:
			# 	print(e)
			# finally:
			# 	session.close()
		
			# get video by id and calculate statistic
			session = DbBase.get_session(True)

			ads_statistic = []
			for video_id in video_ids:
				# get all logs of the video
				log_list = session.query(l) \
					.filter(and_(
						l.videoId == video_id, 
						l.createdAt >= from_time, 
						l.createdAt < to_time)
					).all()

				# df = pd.DataFrame(columns=['videoId', 'gender', 'age', 'happy', 'createdAt'])
				df_data = []
				# calculate statistic
				for log in log_list:
					videoId = log.videoId
					# 'gender': np.zeros(2),
					gender = json.loads(log.gender)
					# 'age': np.zeros(5),
					age = json.loads(log.age)
					# 'happy': [
					# 	# gender
					# 	np.zeros(2),
					# 	# age
					# 	np.zeros(5),
					# ]
					happy = json.loads(log.happy)
					df_data.append({
						'group': statistic_group,
						'videoId': videoId,
						'gender': gender,
						'age': age,
						'happy': happy,
						'createdAt': log.createdAt.strftime("%Y-%m-%d %H:%M:%S"),
					})

				df = pd.DataFrame(df_data)
				ads_statistic.append({
					'group': statistic_group,
					'videoId': videoId,
					'gender': str([sum(i) for i in zip(*df.gender)]),
					'age': str([sum(i) for i in zip(*df['age'])]),
					'happy': str([[sum(i) for i in zip(*j)] for j in zip(*df.happy)]),
				})

			if (len(ads_statistic) > 0):
				df = pd.DataFrame(ads_statistic)

				# soft delete old statistic (if any)
				DbBase.soft_delete(Statistic, condition = ( 
					Statistic.group == statistic_group, 
					Statistic.videoId.in_(df.videoId.unique().tolist()), 
				))

				# FOR DEBUG
				print(df)
				return DbBase.bulk_insert(Statistic, df.columns, df.values)

		except Exception as e:
			traceback.print_exc()
	
		finally:
			if (session):
				session.close()
		
		return None

		

	