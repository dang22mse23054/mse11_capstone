import os

class CacheService():
	def __init__(self, file_path, val_type = 'string'):
		self.file_path = file_path
		self.val_type = val_type

	def is_existed(self):
		return os.path.isfile(self.file_path)

	def set(self, value):
		if (value is not None):
			f = open(self.file_path, "w+")
			f.write(','.join(value) if type(value).__name__ == 'list' else value)
		
	def get(self):
		default_val = [] if self.val_type == 'list' else None
		try:
			if (self.is_existed()):
				f = open(self.file_path, "r")
				try:
					val = f.read()
					return val.split(',') if self.val_type == 'list' else val
				finally:
					f.close()
		except Exception as e:
			print(e)

		return default_val

	def delete(self):
		if (self.is_existed()):
			os.remove(self.file_path)
