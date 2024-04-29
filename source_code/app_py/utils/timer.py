from dateutil import tz
from datetime import datetime, timedelta, date, time

TIME_ZONE = "UTC"
# TIME_ZONE = "Asia/Ho_Chi_Minh"

class Timer():
	def __init__(self, label=None, is_enabled = True):
		self.start_time = None
		self.stop_time = None
		self.label = label
		self.is_enabled = is_enabled
  
	def start(self, label=None):
		self.start_time = datetime.now()
		self.label = label
  
	def stop(self):
		self.stop_time = datetime.now()
		if (self.is_enabled):
			print(f"{self.label}: {(self.stop_time - self.start_time).total_seconds()} seconds")
  
class TimeUtils:
	
	@classmethod
	def now(cls, timezone = TIME_ZONE):
		return datetime.now(tz.gettz(timezone))

	@classmethod
	def get_start_of_date(cls, input_time = datetime.now(), timezone = TIME_ZONE):
		return datetime.combine(input_time, time.min, tz.gettz(timezone))
	
	@classmethod
	def get_start_of_hours(cls, input_time = datetime.now(), hours = 0, timezone = None):
		output = (input_time + timedelta(hours = hours)).replace(minute=0, second=0, microsecond=0)
		if (timezone is not None):
			return output.astimezone(tz.gettz(timezone))
		return output
	
	@classmethod
	def get_start_of_month(cls, input_time = datetime.now(), timezone = TIME_ZONE):
		# return UtilSerivce.convert_tz(datetime.combine(input_time, time.max), timezone)
		return datetime.combine(input_time, time.min, tz.gettz(timezone)).replace(day=1)
	
	@classmethod
	def get_end_of_date(cls, input_time = datetime.now(), timezone = TIME_ZONE):
		return datetime.combine(input_time, time.max, tz.gettz(timezone))
	