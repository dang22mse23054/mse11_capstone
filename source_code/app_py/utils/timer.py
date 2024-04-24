from datetime import datetime

class Timer():
	def __init__(self, label=None):
		self.start_time = None
		self.stop_time = None
		self.label = label
  
	def start(self, label=None):
		self.start_time = datetime.now()
		self.label = label
  
	def stop(self):
		self.stop_time = datetime.now()
		print(f"{self.label}: {(self.stop_time - self.start_time).total_seconds()} seconds")
  