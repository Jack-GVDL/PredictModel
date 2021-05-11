from typing import *
import datetime
from ..DataChain import *
from .Data_Raw import DataKeyLib_Raw


class DataConversion_TimeRound(DataConversion):

	def __init__(self):
		super().__init__()

		# data
		self._name					= "TimeRounding"
		self._map_list:	List[int] 	= [-1, -1, -1, -1]
		self._is_ready:	bool		= False

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# ...

	# Protected
	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		# check
		if len(key_src_list) != 1:
			return False
		if key_src_list[0] != key_dst:
			return False

		# find the date and time key in src and dst respectively
		key_src: 		DataKey = key_src_list[0]
		self._is_ready 			= False

		# format: [src_date, src_time, dst_date, dst_time]
		self._map_list[0] = key_src.getKeyIndex_Key(DataKeyLib_Raw.key_date)
		self._map_list[1] = key_src.getKeyIndex_Key(DataKeyLib_Raw.key_time)
		self._map_list[2] = key_dst.getKeyIndex_Key(DataKeyLib_Raw.key_date)
		self._map_list[3] = key_dst.getKeyIndex_Key(DataKeyLib_Raw.key_time)

		# check if mapping is valid or not
		if self._map_list[0] == -1 or self._map_list[1] == -1:
			return False
		self._is_ready = True

		# set to key_src and key_dst
		self._key_src_list 	= key_src_list.copy()
		self._key_dst		= key_dst

		return True

	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# check
		if not self._is_ready:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		if not src_list:
			log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS)
			return True

		# foreach data
		count_input: 	int = 0
		count_output:	int = 0

		for src in src_list:
			for data in src:
				self._roundTime_(data, dst)
			count_input += len(src)

		# stat
		count_output += len(dst)

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, count_input, count_output)

		return True

	# TODO: now is all round to nearest 10 minute
	def _roundTime_(self, target: DataBase, data_list: DataList) -> None:
		# get date and time
		date: List[int] = target[self._map_list[0]]
		time: List[int] = target[self._map_list[1]]

		target_time = datetime.datetime(
			year=date[0], month=date[1], day=date[2],
			hour=time[0], minute=time[1], second=0
		)

		# actual rounding
		if time[1] % 10 > 5:
			target_time = target_time + datetime.timedelta(minutes=10 - time[1] % 10)
		else:
			target_time = target_time - datetime.timedelta(minutes=time[1] % 10)

		# create data
		data: DataBase = data_list.copyData(target)

		# set time
		data[self._map_list[2]] = [target_time.year, target_time.month, target_time.day]
		data[self._map_list[3]] = [target_time.hour, target_time.minute, target_time.second]


class DataConversion_TimeRange(DataConversion):

	def __init__(self):
		super().__init__()

		# data
		self._name					= "TimeRange_V2"
		self._map_list:	List[int] 	= [-1, -1]
		self._is_ready:	bool		= False

		self._time_range:		List[int]	= [0, 1]
		self._interval:			int = 10

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	def setRange(self, time_range: List[int], interval: int) -> None:
		self._time_range	= time_range
		self._interval 		= interval

	# Protected
	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		# check
		if len(key_src_list) != 1:
			return False
		if key_src_list[0] != key_dst:
			return False

		# find the date and time key in src and dst respectively
		key_src: 		DataKey = key_src_list[0]
		self._is_ready 			= False

		self._map_list[0] = key_src.getKeyIndex_Key(DataKeyLib_Raw.key_date)
		self._map_list[1] = key_src.getKeyIndex_Key(DataKeyLib_Raw.key_time)

		# check if mapping is valid or not
		if self._map_list[0] == -1 or self._map_list[1] == -1:
			return False
		self._is_ready = True

		# set to key_src and key_dst
		self._key_src_list 	= key_src_list.copy()
		self._key_dst		= key_dst

		return True

	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# check
		if not self._is_ready:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		if not src_list:
			log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS)
			return True

		# foreach data
		count_input: 	int = 0
		count_output:	int = 0

		for src in src_list:
			for data in src:
				self._rangeTime_(data, dst)
			count_input += len(src)

		# stat
		count_output += len(dst)

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, count_input, count_output)

		return True

	def _rangeTime_(self, target: DataBase, data_list: DataList) -> None:
		# get date and time
		date: List[int] = target[self._map_list[0]]
		time: List[int] = target[self._map_list[1]]

		time_base = datetime.datetime(
			year=date[0], month=date[1], day=date[2],
			hour=time[0], minute=time[1], second=0
		)

		# for range of time slice [T1, T2]
		# based on a reference time T
		# generate a range of time starting from T1 to T2 (T2 is excluded) with interval
		#
		# example
		# T = 10:30, T1 = -2, T2 = 0, interval = 10 minute = 10 * 60 second
		#
		# then the resultant time slice will be
		# [10:10, 10:20]
		for index in range(self._time_range[0], self._time_range[1]):
			time_cur = time_base + datetime.timedelta(seconds=self._interval * index)

			data: DataBase = data_list.createData()
			data.setData(target, is_inplace=False)

			data[self._map_list[0]] = [time_cur.year, time_cur.month, time_cur.day]
			data[self._map_list[1]] = [time_cur.hour, time_cur.minute, time_cur.second]
