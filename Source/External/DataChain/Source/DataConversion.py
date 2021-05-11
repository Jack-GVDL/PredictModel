"""
Data Conversion
- extend
- map
- filter
- sort
- unique
- replace (conditional map / partial join)
"""


from typing import *
import time
from functools import cmp_to_key
from .DataList import DataKey
from .DataList import DataBase
from .DataList import DataList


class DataUpdateLog:

	class State:
		FAIL:		int = 1
		SUCCESS:	int = 2

	def __init__(self):
		super().__init__()

		# data
		self.src_list:		List[DataList]	= []
		self.dst:			DataList		= None

		self.conversion:	Any				= None
		self.state:			int				= -1

		# number of data that processed by this conversion
		# the definition of "data that is processed" is depended on each conversion
		# and
		# therefore cannot use this value to determine the length of data in src or dst
		#
		# e.g.
		# for loader, data processed: data loaded => len(dst)
		# for dumper, data processed: data dumped => count = 0; for src in src_list: count += len(src)
		self.size_input:	int				= 0
		self.size_output:	int				= 0

		# miscellaneous
		self.data: 			Dict 			= {}

		# time
		self._time_start:	int				= 0
		self._time_end:		int				= 0
		self.duration:		float			= 0

		# operation
		# ...

	def __del__(self) -> None:
		return

	# Property
	# ...

	# Operation
	def log(
		self,
		src_list: 		List[DataList],
		dst: 			DataList,
		conversion:		Any,
		state: 			int,
		size_input: 	int				= 0,
		size_output:	int				= 0,
		data: 			Dict			= dict({})) -> None:

		self.src_list 		= src_list
		self.dst			= dst
		self.conversion		= conversion
		self.state			= state
		self.size_input		= size_input
		self.size_output	= size_output
		self.data			= data

	def start(self, time_start: time.time) -> None:
		self._time_start = time_start

	def end(self, time_end: time.time) -> None:
		self._time_end = time_end
		self.duration = self._time_end - self._time_start

	# Protected
	# ...


class DataConversion:

	def __init__(self):
		super().__init__()

		# data
		self._name:			str				= "unknown"
		self._key_dst:		DataKey			= None
		self._key_src_list:	List[DataKey]	= []

		# operation
		# ...

	def __del__(self):
		return

	# Property
	@property
	def name(self) -> str:
		return self._name

	@property
	def key_src_list(self) -> List[DataKey]:
		return self._key_src_list

	@property
	def key_dst(self) -> DataKey:
		return self._key_dst

	# Operation
	def setType(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		return self._setType_(key_src_list, key_dst)

	def convert(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# check
		if not self._checkType_Src_(src_list):
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL, 0)
			return False

		if not self._checkType_Dst_(dst):
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL, 0)
			return False

		# actual conversion
		if not self._convert_(src_list, dst, log):
			return False
		return True

	# Protected
	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		raise RuntimeError("src_key_list and key_dst cannot be configured")

	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		raise NotImplementedError

	def _checkType_Src_(self, src_list: List[DataList]) -> bool:
		# backup
		# if len(src_list) != len(self._key_src_list):
		# 	return False
		# for index in range(len(self._key_src_list)):
		# 	if not src_list[index].data_key.checkType(self._key_src_list[index]):
		# 		continue
		# 	return False

		# key_src_list can be empty
		# check if key_src_list is empty or not
		# if not self._key_src_list:
		# 	return False

		# broadcasting
		# broadcasting will take effect if the len(self._key_src_list) < len(src_list)
		index_src: int = 0

		# foreach key (src_list)
		for src in src_list:

			if self._key_src_list[index_src] == src.data_key:
				index_src = (index_src + 1) % len(self._key_src_list)
				continue
			return False

		return True

	def _checkType_Dst_(self, dst: DataList) -> bool:
		# backup
		# if not dst.checkType(self._key_dst):
		# 	return False

		if self._key_dst is None:
			return False

		if dst.data_key != self._key_dst:
			return False
		return True


class DataConversion_Extend(DataConversion):

	def __init__(self):
		super().__init__()

		# data
		self._name = "Extend"

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# ...

	# Protected
	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# extend
		count_input: 	int = 0
		count_output:	int = 0

		# foreach DataList
		for src in src_list:
			dst._data_list.extend(src._data_list)
			count_input += len(src)

		# stat
		count_output += len(dst)

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, count_input, count_output)

		return True

	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		if len(key_src_list) != 1:
			return False
		if key_src_list[0] != key_dst:
			return False

		self._key_src_list 	= key_src_list
		self._key_dst		= key_dst
		return True


class DataConversion_Map(DataConversion):

	def __init__(self):
		super().__init__()

		# data
		self._name = "Map"

		# TODO: add "axis" feature
		# self._axis: int = -1

		# length of map_list should be the same as length of dst data_key
		self._map_list:		List[int] 	= []
		self._is_mapped:	bool		= False

		# operation
		# ...
	
	def __del__(self):
		return

	# Operation
	# ...

	# Protected
	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		if len(key_src_list) != 1:
			return False

		self._key_src_list	= key_src_list
		self._key_dst		= key_dst

		self._computeMapping_()
		return True

	def _computeMapping_(self) -> None:
		# assumed: key_src and key_dst must be valid
		# reset map list
		self._map_list.clear()
		for _ in range(len(self._key_dst.key_list)):
			self._map_list.append(-1)

		# foreach dst key, search foreach src key
		for dst_index, dst_key in enumerate(self._key_dst.key_list):
			for src_index, src_key in enumerate(self._key_src_list[0].key_list):

				if dst_key != src_key:
					continue
				self._map_list[dst_index] = src_index
				break

	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# foreach DataList, then foreach DataBase
		count_input: 	int = 0
		count_output:	int = 0

		for src in src_list:
			for data in src:
				self._convertData_(data, dst)
			count_input += len(src)

		# stat
		count_output += len(dst)

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, count_input, count_output)

		return True

	def _convertData_(self, src: DataBase, data_list: DataList) -> None:
		# allocate space inside data_list
		dst = data_list.createData()

		for dst_index, src_index in enumerate(self._map_list):
			if src_index == -1:
				continue
			dst[dst_index] = src[src_index]


class DataConversion_Filter(DataConversion):

	def __init__(self, func_compare: Callable[[DataBase], bool]):
		super().__init__()

		# data
		self._name = "Filter"
		self.func_compare: Callable[[DataBase], bool] = None

		# operation
		self.func_compare = func_compare

	def __del__(self):
		return

	# Operation
	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		if self.func_compare is None:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		# foreach DataList, then foreach DataBase
		count_input:	int = 0
		count_output:	int = 0

		for src in src_list:
			for data in src:

				if not self.func_compare(data):
					continue

				temp: DataBase = dst.createData()
				temp.setData(data)

			# stat
			count_input += len(src)

		# stat
		count_output += len(dst)

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, count_input, count_output)

		return True

	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		if len(key_src_list) != 0:
			return False
		if key_src_list[0] != key_dst:
			return False

		self._key_src_list 	= key_src_list
		self._key_dst		= key_dst
		return True


class DataConversion_Sort(DataConversion):

	def __init__(self):
		super().__init__()
		
		# data
		self._name = "Sort"
		self._key_index: List[int] = []
		
		# operation
		# ...
		
	def __del__(self):
		return

	# Operation
	def setTargetKey(self, key_list: List[DataKey]) -> bool:
		# self._key_dst and self._key_src_list[0] should be the same
		if self._key_dst is None:
			return False

		# foreach key, get its index
		for key in key_list:
			index: int = self._key_dst.getKeyIndex_Key(key)

			# index found
			if index != -1:
				self._key_index.append(index)
				continue

			# index not found
			return False

		return True

	# Protected
	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# check
		if not self._key_index:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		# add all the DataList to dst
		count_input: 	int = 0
		count_output:	int = 0

		for src in src_list:
			dst._data_list.extend(src._data_list)
			count_input += len(src)

		# stat
		count_output = len(dst)

		# sort
		# reference
		# - https://stackoverflow.com/questions/5213033/sort-a-list-of-lists-with-a-custom-compare-function
		dst._data_list.sort(key=cmp_to_key(self._compare_))

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, count_input, count_output)

		return True

	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		if len(key_src_list) != 1:
			return False
		if key_src_list[0] != key_dst:
			return False

		self._key_src_list 	= key_src_list
		self._key_dst		= key_dst
		return True

	def _compare_(self, item_1: List[Any], item_2: List[Any]) -> int:
		for index in self._key_index:
			if item_1[index] < item_2[index]:
				return -1
			elif item_1[index] > item_2[index]:
				return 1
		return 0


class DataConversion_Unique(DataConversion):

	def __init__(self):
		super().__init__()

		# data
		self._name = "Unique"
		self._key_index: List[int] = []

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	def setTargetKey(self, key_list: List[DataKey]) -> bool:
		# self._key_dst and self._key_src_list[0] should be the same
		if self._key_dst is None:
			return False

		# foreach key, get its index
		for key in key_list:
			index: int = self._key_dst.getKeyIndex_Key(key)

			# index found
			if index != -1:
				self._key_index.append(index)
				continue

			# index not found
			return False

		return True

	# Protected
	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# check
		if not self._key_index:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		# ----- sort -----
		temp = DataList()
		for src in src_list:
			temp._data_list.extend(src._data_list)

		# check if temp.data_list is empty or not
		# if empty, then do nothing
		if not temp:
			log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS)
			return True

		# actual sorting
		temp._data_list.sort(key=cmp_to_key(self._compare_))

		# ----- unique -----
		# setup
		prev_data = temp[0]
		data_new = dst.createData()
		data_new.setData(prev_data)

		# foreach remaining data
		for index in range(1, len(temp)):
			data: DataBase = temp[index]

			# same
			if self._compare_(data.data, prev_data.data) == 0:
				continue

			# new
			prev_data = data
			data_new = dst.createData()
			data_new.setData(prev_data)

		# stat
		count_input 	= len(temp)
		count_output	= len(dst)

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, count_input, count_output)

		return True

	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		if len(key_src_list) != 1:
			return False
		if key_src_list[0] != key_dst:
			return False

		self._key_src_list 	= key_src_list
		self._key_dst		= key_dst
		return True

	def _compare_(self, item_1: List[Any], item_2: List[Any]) -> int:
		for index in self._key_index:
			if item_1[index] < item_2[index]:
				return -1
			elif item_1[index] > item_2[index]:
				return 1
		return 0


class DataConversion_Replace(DataConversion):

	def __init__(self):
		super().__init__()

		# data
		self._name = "Replace"

		self._key_compare:	List[int] = []
		self._key_map:		List[int] = []

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	def setCompareKey(self, key_list: List[DataKey]) -> bool:
		# assumed: self._key_dst and self._key_src_list[any_index] should be the same
		if self._key_dst is None:
			return False

		# foreach key, get its index
		self._key_compare.clear()
		for key in key_list:
			index: int = self._key_dst.getKeyIndex_Key(key)

			# index found
			if index != -1:
				self._key_compare.append(index)
				continue

			# index not found
			return False

		return True

	def setMapKey(self, key_list: List[DataKey]) -> bool:
		# assumed: self._key_dst and self._key_src_list[any_index] should be the same
		if self._key_dst is None:
			return False

		# foreach key, get its index
		self._key_map.clear()
		for key in key_list:
			index: int = self._key_dst.getKeyIndex_Key(key)

			# index found
			if index != -1:
				self._key_map.append(index)
				continue

			# index not found
			return False

		return True

	# Protected
	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# check
		if not self._key_compare or not self._key_map:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		# ----- sort -----
		# the first item in the src_list is the target list for replacement
		for data in src_list[0]:
			data_new: DataBase = dst.createData()
			data_new.setData(data)

		# merge src_list[1:]
		temp = DataList()
		for i in range(1, len(src_list)):
			temp._data_list.extend(src_list[i]._data_list)

		# check if temp.data_list is empty or not
		# if empty, then do nothing (no replacement but the dst_list is correct)
		if not temp:
			log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS)
			return True

		# actual sorting
		temp._data_list.sort(key=cmp_to_key(self._compare_))

		# ----- replace -----
		for data in dst:
			for data_temp in temp:

				# compare
				#
				# data < data_temp: the target data may be in later item in temp
				# data > data_temp: the target data does not exist in temp
				# data == data_temp: data_temp is the target data
				#
				# be noted that the comparison is partial compare, not full compare
				ret: int = self._compare_(data.data, data_temp.data)
				if ret == -1:
					continue
				if ret == 1:
					break

				# replace
				for index in self._key_map:
					data[index] = data_temp[index]

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, len(src_list[0]), len(dst))

		return True

	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		if len(key_src_list) != 1:
			return False

		for key_src in key_src_list:
			if key_src == key_dst:
				continue
			return False

		self._key_src_list 	= key_src_list
		self._key_dst		= key_dst
		return True

	def _compare_(self, item_1: List[Any], item_2: List[Any]) -> int:
		for index in self._key_compare:
			if item_1[index] < item_2[index]:
				return -1
			elif item_1[index] > item_2[index]:
				return 1
		return 0


# ----- group -----
class DataKey_Group(DataKey):

	# data
	_data_key: DataKey = None

	@classmethod
	def getDataKey(cls) -> DataKey:
		if cls._data_key is None:
			cls._data_key = DataKey_Group()
		return cls._data_key

	def __init__(self):
		super().__init__()

		# data
		# ...

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# ...


class DataConversion_Group(DataConversion):

	def __init__(self):
		super().__init__()

		# data
		self._name				= "Group"
		self._dst 				= DataKey()
		self._key_index: int 	= -1

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	def setTargetKey(self, key: DataKey) -> bool:
		if not self._key_src_list:
			return False

		index: int = self._key_dst.getKeyIndex_Key(key)
		if index == -1:
			return False

		self._key_index = index

		# set the key_dst[1]
		# which is the key for group separation
		self._key_dst[1] = self._key_src_list[0][self._key_index]

		return True

	# Protected
	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# check
		if self._key_index == -1:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		# config
		src: DataList = src_list[0]
		dst.data_key = self._dst

		# ----- sort -----
		# need to sort the list first
		temp = DataList()
		temp.data_key = src.data_key
		temp._data_list.extend(src._data_list)

		# check if temp.data_list is empty or not
		# if empty, then do nothing
		if not temp:
			log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, 0)
			return True

		# actual sorting
		temp._data_list.sort(key=lambda x: x[self._key_index])

		# ----- group -----
		# setup
		prev_data 	= temp[0]
		cur_list	= self._createGroup_(prev_data[self._key_index], dst)
		data_new	= cur_list.createData()
		data_new.setData(prev_data)

		# foreach remaining data
		for index in range(1, len(temp)):
			data: DataBase = temp[index]

			# same group
			if data[self._key_index] == prev_data[self._key_index]:
				prev_data = data
				data_new = cur_list.createData()
				data_new.setData(prev_data)
				continue

			# new group
			prev_data = data
			cur_list = self._createGroup_(prev_data[self._key_index], dst)
			data_new = cur_list.createData()
			data_new.setData(prev_data)

		# stat
		count_input 	= len(temp)
		count_output	= len(dst)

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, count_input, count_output)
		return True

	def _createGroup_(self, key: Any, target_list: DataList) -> DataList:
		# create data list
		data_list = DataList(self._dst)

		# create data group
		data_group = target_list.createData()
		data_group[0]	= None  # identifier key
		data_group[1] 	= key
		data_group[2]	= data_list

		return data_list

	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		if key_dst is not None:
			return False
		if len(key_src_list) != 1:
			return False

		# key_dst will be computed by self
		key_src = key_src_list[0]

		key_dst = DataKey()
		key_dst.addDataKey(DataKey_Group.getDataKey())  # add identifier
		key_dst.addDataKey(None)  						# will be computed later
		key_dst.addDataKey(key_src)

		self._key_src_list	= key_src_list
		self._key_dst		= key_dst
		return True


class DataConversion_Ungroup(DataConversion):

	def __init__(self):
		super().__init__()
		
		# data
		self._name		= "Ungroup"
		self._key: int 	= -1
		
		# operation
		# ...
		
	def __del__(self):
		return
		
	# Operation
	# ...

	# Protected
	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# check stage do not verify the length of src_list
		if len(src_list) != 1:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		# merge all group
		count_input: 	int = 0
		count_output:	int = 0

		src: DataList = src_list[0]
		for data in src._data_list:

			group: DataList = data[2]
			dst._data_list.extend(group._data_list)
			count_input += len(group)

		# stat
		count_output += len(dst)

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, count_input, count_output)
		return True

	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		# check src and dst
		if key_dst is not None:
			return False
		if len(key_src_list) != 1:
			return False
		if key_src_list[0].key_list:
			return False

		# verify the identifier key (DataKey_Group)
		if key_src_list[0][0] != DataKey_Group.getDataKey():
			return False

		# key_dst will be computed by self
		key_src = key_src_list[0]
		key_dst = key_src[2]

		self._key_src_list	= key_src_list
		self._key_dst		= key_dst
		return True
