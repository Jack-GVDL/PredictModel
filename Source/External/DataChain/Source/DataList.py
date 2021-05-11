from typing import *
import datetime
from .DataKey import DataKey


class SnapshotControl:

	def __init__(self):
		super().__init__()

		# data
		self.snapshot:	int = 0

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def createSnapshot(self) -> int:
		# therefore, the first snapshot published is 1
		self.snapshot += 1
		return self.snapshot

	# Protected
	# ...



class DataBase:

	def __init__(self, index: int, parent: Any):
		super().__init__()

		# data
		self._parent:	Any	= None
		self._index: 	int	= -1

		# operation
		self._parent = parent
		self._index	= index

	def __del__(self):
		return

	# Property
	@property
	def data(self) -> List[Any]:
		return self._parent._data_list[self._index]

	@property
	def data_key(self) -> DataKey:
		return self._parent._data_key

	# Operation
	def setData(self, data: Any, is_inplace: bool = False) -> bool:
		# TODO: need assert check
		# ...

		# set or copy
		if is_inplace:
			self._parent._data_list[self._index] = data._parent._data_list[data._index]
		else:
			self._parent._data_list[self._index] = data._parent._data_list[data._index].copy()

		return True

	def setContent(self, content: List[Any], is_inplace) -> bool:
		# TODO: need assert check
		# ...

		# set or copy
		if is_inplace:
			self._parent._data_list[self._index] = content
		else:
			self._parent._data_list[self._index] = content.copy()

		return True

	# Protected
	# ...

	# Operator Overloading
	def __getitem__(self, index: int) -> Any:
		assert self._parent is not None
		return self._parent._data_list[self._index][index]

	def __setitem__(self, index: int, value: Any) -> None:
		assert self._parent is not None
		self._parent._data_list[self._index][index] = value

	def __len__(self):
		return len(self._parent._data_key.key_list)

	def __str__(self) -> str:
		assert self._parent is not None
		data = self._parent._data_list[self._index]

		content: str = ""
		for index, d in enumerate(data):
			content += f"[{index}]: {d}; "
		return content


class DataList:

	def __init__(self, data_key: DataKey = None, name: str = ""):
		super().__init__()

		# ----- data -----
		# data info
		self.name: 				str 			= ""

		self._data_key:			DataKey			= None
		self._data_list: 		List[List[Any]] = []

		# time snap / version
		self.snapshot_control:	SnapshotControl = None  # controller that distribute snapshot / version number

		# format: [YYYY, MM, DD, hh, mm, ss]
		# self.time_snap:		List[int]		= [0, 0, 0, 0, 0, 0]
		self.snapshot:			int				= -1

		# for static data, it is always up to date
		# for volatile data, it is always need update (itself keeps on refreshing)
		self.is_static:			bool			= False
		self.is_volatile:		bool			= False

		# ----- operation -----
		self.name		= name
		self._data_key	= data_key

	def __del__(self):
		return

	# Property
	@property
	def data_key(self) -> DataKey:
		return self._data_key

	# @property
	# def data_list(self):
	# 	return self._data_list

	@data_key.setter
	def data_key(self, key: DataKey) -> None:
		self._data_key = key

	# Operation
	def reset(self) -> bool:
		self._data_list.clear()
		return True

	def createData(self) -> DataBase:
		# check
		if self._data_key is None:
			return None

		# allocate space
		data: List[Any] = [None for _ in range(len(self._data_key.key_list))]
		self._data_list.append(data)

		# create DataBase
		data_base = DataBase(len(self._data_list) - 1, self)
		return data_base

	def copyData(self, src: DataBase) -> DataBase:
		# check
		if self._data_key != src.data_key:
			return None

		# allocate space
		data: List[Any] = []
		data.extend(src.data)
		self._data_list.append(data)

		# create DataBase
		data_base = DataBase(len(self._data_list) - 1, self)
		return data_base

	def clearData(self) -> bool:
		self._data_list.clear()
		self._markUpdate_()
		return True

	def markUpdate(self) -> None:
		self._markUpdate_()

	# Protected
	def _markUpdate_(self) -> None:
		# backup
		# current_datetime = datetime.datetime.today()
		#
		# self.time_snap = [
		# 	current_datetime.year, current_datetime.month, current_datetime.day,
		# 	current_datetime.hour, current_datetime.minute, current_datetime.second
		# ]

		self.snapshot = self.snapshot_control.createSnapshot()

	# Operator Overload
	def __getitem__(self, index: int) -> DataBase:
		if index < 0 or index >= len(self._data_list):
			raise IndexError
		data_base = DataBase(index, self)
		return data_base

	def __len__(self) -> int:
		return len(self._data_list)
