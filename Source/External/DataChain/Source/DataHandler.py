from typing import *
from .DataList import DataKey
from .DataList import DataBase
from .DataList import DataList
from .DataConversion import DataConversion
from .DataConversion import DataUpdateLog


class DataHandler:

	def __init__(self):
		super().__init__()

		# data
		self._data_key: DataKey = None

	# operation
	# ...

	def __del__(self):
		return

	# Property
	@property
	def data_key(self) -> DataKey:
		return self._data_key

	@data_key.setter
	def data_key(self, key: DataKey) -> None:
		self._data_key = key

	# Operation
	def load(self, data_list: DataList) -> bool:
		# check
		if data_list is None:
			return False

		# actual load
		if not self._load_(data_list):
			return False

		return True

	def dump(self, data_list: DataList) -> bool:
		# check
		# if self.data_node is None:
		# 	return False
		if data_list is None:
			return False

		# actual load
		if not self._dump_(data_list):
			return False

		return True

	# Protected
	def _load_(self, data_list: DataList) -> bool:
		raise NotImplementedError

	def _dump_(self, data_list: DataList) -> bool:
		raise NotImplementedError


class DataConversion_Loader(DataConversion):

	def __init__(self, handler: DataHandler = None, key: DataKey = None):
		super().__init__()

		# data
		self._name					= "Loader"
		self._handler: DataHandler 	= None

		# operation
		self._handler = handler
		self.setType([], key)

	def __del__(self):
		return

	# Property
	@property
	def handler(self):
		return self._handler

	# @handler.setter
	# def handler(self, h: DataHandler) -> None:
	# 	self._handler = h

	# Operation
	def setHandler(self, h: DataHandler) -> bool:
		self._handler = h
		self._setType_([], h.data_key)
		return True

	# Protected
	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# check
		if self._handler is None:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		# actual load
		result: bool = self._handler.load(dst)
		if not result:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, 0, len(dst))

		# TODO: this operation should be configurable
		# mark the dst as "static" once the data is loaded to dst
		dst.is_static = True

		return True

	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		# key_src_list should be empty
		if key_src_list:
			return False

		self._key_src_list 	= key_src_list
		self._key_dst 		= key_dst
		return True


class DataConversion_Dumper(DataConversion):

	def __init__(self, handler: DataHandler = None, key: DataKey = None):
		super().__init__()

		# data
		self._name					= "Dumper"
		self._handler: DataHandler 	= None

		# operation
		self._handler = handler
		self.setType([key], key)

	def __del__(self):
		return

	# Property
	@property
	def handler(self):
		return self._handler

	# @handler.setter
	# def handler(self, h: DataHandler) -> None:
	# 	self._handler = h

	# Operation
	def setHandler(self, h: DataHandler) -> bool:
		self._handler = h
		self._setType_([h.data_key], h.data_key)
		return True

	# Protected
	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# check
		if len(src_list) != 1:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		# actual load
		result: bool = self._handler.dump(src_list[0])
		if not result:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, len(src_list[0]), 0)

		return True

	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		if len(key_src_list) != 1:
			return False
		if key_src_list[0] != key_dst:
			return False

		self._key_src_list = key_src_list
		self._key_dst = key_dst
		return True
