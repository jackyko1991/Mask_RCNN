import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from mrcnn.config import Config


class JsonConfig(Config):
	"""Configuration for by reading json files.
	Derives from the base Config class and overrides values specific
	to the json provided.
	"""
	
	def readJson(self,config_json):
		if not os.path.exists(config_json):
			print("Config file not exists, abort")
			exit()

		with open(config_json, 'r') as f:
			config = json.load(f)

		for key, value in config.items():
			# here we implement selection of gpu
			if key == "CUDA_VISIBLE_DEIVCE":
				setattr(self, "CUDA_VISIBLE_DEIVCE", value)
				setattr(self, "GPU_COUNT", len(value))
			else:
				setattr(self, key, value)
		self.display()

	def getConfigValue(self,key):
		return getattr(self,key)