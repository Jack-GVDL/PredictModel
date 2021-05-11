from typing import *
from ..DataChain import *
from .Data_Raw import DataKeyLib_Raw


class DataKeyLib_QAR:
	key_forecast_time:	DataKey	= DataKey("ForecastTime")
	key_edr:			DataKey = DataKey("EDR")
	key_delta_g:		DataKey = DataKey("DeltaG")


class DataKey_QAR(DataKey):

	def __init__(self):
		super().__init__()
		
		# data
		self._name = "QAR"

		# index
		self.date:			int = -1
		self.time:			int = -1
		self.forecast_time:	int = -1
		self.longitude:		int = -1
		self.latitude:		int = -1
		self.height:		int = -1
		self.edr:			int = -1
		self.delta_g:		int = -1

		# key
		self.key_date:				DataKey = DataKeyLib_Raw.key_date
		self.key_time:				DataKey = DataKeyLib_Raw.key_time
		self.key_forecast_time:		DataKey = DataKeyLib_QAR.key_forecast_time
		self.key_longitude:			DataKey = DataKeyLib_Raw.key_longitude
		self.key_latitude:			DataKey = DataKeyLib_Raw.key_latitude
		self.key_height:			DataKey = DataKeyLib_Raw.key_height
		self.key_edr:				DataKey = DataKeyLib_QAR.key_edr
		self.key_delta_g:			DataKey = DataKeyLib_QAR.key_delta_g

		# operation
		# add key
		self.addDataKey(self.key_date)
		self.addDataKey(self.key_time)
		self.addDataKey(self.key_forecast_time)
		self.addDataKey(self.key_longitude)
		self.addDataKey(self.key_latitude)
		self.addDataKey(self.key_height)
		self.addDataKey(self.key_edr)
		self.addDataKey(self.key_delta_g)

		# get index
		self.date			= self.getKeyIndex_Key(self.key_date)
		self.time			= self.getKeyIndex_Key(self.key_time)
		self.forecast_time	= self.getKeyIndex_Key(self.key_forecast_time)
		self.longitude		= self.getKeyIndex_Key(self.key_longitude)
		self.latitude		= self.getKeyIndex_Key(self.key_latitude)
		self.height			= self.getKeyIndex_Key(self.key_height)
		self.edr			= self.getKeyIndex_Key(self.key_edr)
		self.delta_g		= self.getKeyIndex_Key(self.key_delta_g)

	def __del__(self):
		return
		
	# Property
	# ...
		
	# Operation
	# ...
	
	# Protected
	# ...


class DataHandler_Text_QAR(DataHandler):

	# height level table
	table_height: List[int] = [
		14000,
		18000,
		24000,
		30000,
		34000,
		39000
	]

	def __init__(self):
		super().__init__()

		# data
		self._data_key	= DataKey_QAR()
		self.file_text	= File_Text()

		# operation
		# ...

	def __del__(self):
		return

	# Property
	@property
	def data_key(self) -> DataKey:
		return self._data_key

	@data_key.setter
	def data_key(self, key: DataKey) -> DataKey:
		raise RuntimeError

	# Operation
	# ...

	# Protected
	def _load_(self, data_list: DataList) -> bool:
		# CHECK
		if self.file_text is None:
			return False

		# load from plain text
		if not self.file_text.load():
			return False

		# foreach item (Data_QAR)
		for item in self.file_text.data:
			self._loadSingleData_(item, data_list)

		return True

	def _dump_(self, data_list: DataList) -> bool:
		raise RuntimeError

	"""
	[0]		Date YYYYMMDDHHMMSS
	[1]		Forecast time: 0 means 12 hr - 21 hr, 1 means 24 hr - 33 hr, 2 means 36 hr - 45 hr, 3 means 48 hr - 57 hr, 4 means 60 hr - 69 hr
	[2]		Latitude: latitude = 90 - (lat_grid) * 0.125
	[3]		Longitude: longitude = -180 + lon_grid * 0.125
	[4]		Height: 1 for FL140, 2 for 180, 3 for 240, 4 for 300, 5 for 340, 6 for 390 and above
	[5]		EDR
	[6]		Delta-g
	[7-10]	ignore
	[11]	TI2
	[12]	TI3
	[13]	TI4
	[14]	VWS
	[15]	DTI
	[16...]	ignore first
	"""
	def _loadSingleData_(self, s: str, data_list: DataList) -> None:
		# CONFIG
		# file data
		string_list: List[str] = s.split()
		if len(string_list) < 7:
			return

		# data_key
		data_key: DataKey_QAR = self._data_key

		# create object
		# TODO: assume creation must be success
		data = data_list.createData()

		# direct conversion
		try:
			data[data_key.date]			= self._convertDate_(string_list[0])
			data[data_key.time]			= self._convertTime_(string_list[0])

			data[data_key.forecast_time]	= int(string_list[1])
			data[data_key.latitude]			= 90 - 0.125 * float(string_list[2])
			data[data_key.longitude]		= -180 + 0.125 * float(string_list[3])
			data[data_key.edr]				= float(string_list[5])
			data[data_key.delta_g]			= float(string_list[6])

		except ValueError:
			return

		# convert height from index to actual value
		try:
			index: int = int(string_list[4])
			data[data_key.height] = self.table_height[index - 1]

		except ValueError:
			return

		# fine tune longitude
		# ensure that value to be positive
		if data[data_key.longitude] < 0:
			data[data_key.longitude] += 360.0

	def _convertDate_(self, s: str) -> List[int]:
		try:
			temp = [int(s[0:4]), int(s[4:6]), int(s[6:8])]
		except ValueError:
			return [0, 0, 0]
		return temp

	def _convertTime_(self, s: str) -> List[int]:
		try:
			temp = [int(s[8:10]), int(s[10:12]), int(s[12:14])]
		except ValueError:
			return [0, 0, 0]
		return temp
