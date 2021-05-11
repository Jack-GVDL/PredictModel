from typing import *
from ..DataChain import *
from .Data_Raw import DataKeyLib_Raw


class DataKey_PIREP(DataKey):

	def __init__(self):
		super().__init__()

		# data
		self._name = "PIREP"

		# key-index
		self.date:				int = -1
		self.time:				int = -1
		self.latitude:			int = -1
		self.longitude:			int = -1
		self.height:			int = -1
		self.pirep_turbulence:	int = -1
		self.pirep_icing:		int = -1

		self.key_date:				DataKey = DataKeyLib_Raw.key_date
		self.key_time:				DataKey = DataKeyLib_Raw.key_time
		self.key_longitude:			DataKey = DataKeyLib_Raw.key_longitude
		self.key_latitude:			DataKey = DataKeyLib_Raw.key_latitude
		self.key_height:			DataKey = DataKeyLib_Raw.key_height
		self.key_pirep_turbulence:	DataKey = DataKeyLib_Raw.key_pirep_turbulence
		self.key_pirep_icing:		DataKey = DataKeyLib_Raw.key_pirep_icing

		# operation
		# add key
		self.addDataKey(self.key_date)
		self.addDataKey(self.key_time)
		self.addDataKey(self.key_longitude)
		self.addDataKey(self.key_latitude)
		self.addDataKey(self.key_height)
		self.addDataKey(self.key_pirep_turbulence)
		self.addDataKey(self.key_pirep_icing)

		# get index
		self.date 				= self.getKeyIndex_Key(self.key_date)
		self.time 				= self.getKeyIndex_Key(self.key_time)
		self.longitude 			= self.getKeyIndex_Key(self.key_longitude)
		self.latitude 			= self.getKeyIndex_Key(self.key_latitude)
		self.height 			= self.getKeyIndex_Key(self.key_height)
		self.pirep_turbulence 	= self.getKeyIndex_Key(self.key_pirep_turbulence)
		self.pirep_icing 		= self.getKeyIndex_Key(self.key_pirep_icing)

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	# ...

	# Protected
	# ...


class DataHandler_Text_PIREP(DataHandler):

	def __init__(self):
		super().__init__()

		# data
		self._data_key	= DataKey_PIREP()
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
		# check
		if self.file_text is None:
			return False

		# load from plain text
		if not self.file_text.load():
			return False

		# for each item (Data_PIREP)
		for item in self.file_text.data:
			self._loadSingleData_(item, data_list)

		return True

	def _dump_(self, data: DataList) -> bool:
		raise RuntimeError

	"""
	Format (tab delimited):
	[0] Event date DDMMYYYY/HHMM (in UTC)
	[1] Turbulence intensity
	[2] Icing intensity (may be absent)
	[3] Flight level (in metres)
	[4] Latitude
	[5] Longitude
	"""
	def _loadSingleData_(self, s: str, data_list: DataList) -> None:
		# setup - file data
		string_list: 	List[str] 	= s.split()
		offset_icing:	int			= 0
		if len(string_list) < 5:
			return

		# setup - data_key
		data_key: DataKey_PIREP = self._data_key

		# create object
		# TODO: assume: creation must be success
		data = data_list.createData()

		# icing may be absent
		try:
			if len(string_list) == 5:
				offset_icing = -1
				data[data_key.pirep_icing] = 0
			else:
				data[data_key.pirep_icing] = int(string_list[2])
		except ValueError:
			return

		# direct conversion
		try:
			data[data_key.date]			= self._convertDate_(string_list[0])
			data[data_key.time]			= self._convertTime_(string_list[0])
			data[data_key.pirep_turbulence]		= int(string_list[1])
			data[data_key.latitude]		= float(string_list[4 + offset_icing])
			data[data_key.longitude]		= float(string_list[5 + offset_icing])

		except ValueError:
			return

		# convert height
		try:
			# from string to int
			height_list: List[str] = string_list[3 + offset_icing].split("-")

			# if the height is a range, get the mean
			if len(height_list) == 1:
				height = int(height_list[0])
			else:
				height = (int(height_list[0]) + int(height_list[1])) / 2

			# convert the value from metric to feet
			# 1 metre = 3.2808398950131 feet
			height = height * 3.2808398950131

			# set to data
			data[data_key.height] = int(height)

		except ValueError:
			return

	def _convertDate_(self, s: str) -> List[int]:
		try:
			temp = [int(s[4:8]), int(s[2:4]), int(s[0:2])]
		except ValueError:
			return [0, 0, 0]
		return temp

	def _convertTime_(self, s: str) -> List[int]:
		try:
			temp = [int(s[9:11]), int(s[11:13]), 0]
		except ValueError:
			return [0, 0, 0]
		return temp
