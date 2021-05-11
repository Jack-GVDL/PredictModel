from typing import *
from ..DataChain import *
from .Data_Raw import DataKeyLib_Raw


class DataKey_Turbulence(DataKey):

	# Static Data
	key_singleton = None

	# Static Function
	# singleton
	@classmethod
	def getDataKey(cls) -> DataKey:
		if cls.key_singleton is None:
			cls.key_singleton = DataKey_Turbulence()
		return cls.key_singleton

	def __init__(self):
		super().__init__()
		
		# data
		self._name = "Turbulence"

		# key-index
		self.date:			int = -1
		self.time:			int = -1
		self.longitude:		int = -1
		self.latitude:		int = -1
		self.height:		int = -1
		self.turbulence:	int = -1

		self.key_date:			DataKey = DataKeyLib_Raw.key_date
		self.key_time:			DataKey = DataKeyLib_Raw.key_time
		self.key_longitude:		DataKey = DataKeyLib_Raw.key_longitude
		self.key_latitude:		DataKey = DataKeyLib_Raw.key_latitude
		self.key_height:		DataKey = DataKeyLib_Raw.key_height
		self.key_turbulence:	DataKey = DataKeyLib_Raw.key_turbulence

		# operation
		# add key
		self.addDataKey(self.key_date)
		self.addDataKey(self.key_time)
		self.addDataKey(self.key_longitude)
		self.addDataKey(self.key_latitude)
		self.addDataKey(self.key_height)
		self.addDataKey(self.key_turbulence)

		# get index
		self.date 			= self.getKeyIndex_Key(self.key_date)
		self.time 			= self.getKeyIndex_Key(self.key_time)
		self.longitude 		= self.getKeyIndex_Key(self.key_longitude)
		self.latitude 		= self.getKeyIndex_Key(self.key_latitude)
		self.height 		= self.getKeyIndex_Key(self.key_height)
		self.turbulence 	= self.getKeyIndex_Key(self.key_turbulence)
		
	def __del__(self):
		return
		
	# Operation
	# ...
