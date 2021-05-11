from typing import *
from .Interface_DictData import Interface_DictData


# Data Structure
class Log_Data(Interface_DictData):

	# Enum
	class DataType:
		NONE:		int = 0
		BOOL:		int = 1
		INT:		int = 2
		FLOAT:		int = 3
		STR:		int = 4
		SIZE_MAX:	int = 5

	class Label:
		ID:			int = 0
		NAME:		int = 1
		DATA:		int = 2
		TYPE:		int = 3
		SIZE_MAX:	int = 4

	def __init__(self):
		super().__init__()

		# data
		self.id_:		int			= -1
		self.name:		str			= "unknown"
		self.data_list: List[Any] 	= []
		self.data_type:	int			= Log_Data.DataType.NONE

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	# ...

	# Interface
	def getDictData(self) -> Dict:
		return {
			Log_Data.Label.ID:		self.id_,
			Log_Data.Label.NAME:	self.name,
			Log_Data.Label.DATA:	self.data_list,
			Log_Data.Label.TYPE:	self.data_type
		}

	def setDictData(self, data: Dict) -> None:
		self.id_		= data[str(Log_Data.Label.ID)]
		self.data_list = data[str(Log_Data.Label.DATA)]
		self.data_type = data[str(Log_Data.Label.TYPE)]

	# Protected
	# ...

	# Operator Overload
	def __len__(self) -> int:
		return len(self.data_list)


class Control_Data(Interface_DictData):

	class Label:
		LOG_DATA_LIST:		int = 0
		INDEX:				int = 1
		SIZE_MAX:			int = 2

	def __init__(self):
		super().__init__()

		# data
		self.log_data_list:	List[Log_Data]	= []
		self.index:			int				= 1

		self._change_list:	Set[int]		= set()

		# operation
		# ...

	def __del__(self):
		return

	# Property
	@property
	def change_list(self) -> List[int]:
		return list(self._change_list)

	# Operation
	def addLog_Data(self, list_: List[Any], type_: int, name: str = "unknown") -> bool:
		# get id of current data
		# avoid using "0"
		id_: int = self.index
		self.index += 1

		# create Log_Data
		log_data = Log_Data()
		log_data.id_ = id_

		log_data.data_list	= list_
		log_data.data_type	= type_
		log_data.name		= name

		# add to list
		self.log_data_list.append(log_data)
		self._change_list.add(id_)

		return True

	def rmLog_Data(self, id_: int) -> bool:
		index: int = self._findIndex_(self.log_data_list, lambda x: x.id_ == id_)
		if index < 0:
			return False

		self.log_data_list.pop(index)
		return True

	def getLog_Data(self, id_: int) -> Log_Data:
		index: int = self._findIndex_(self.log_data_list, lambda x: x.id_ == id_)
		if index < 0:
			return None

		return self.log_data_list[index]

	def setLog_Change(self, id_: int) -> bool:
		index: int = self._findIndex_(self.log_data_list, lambda x: x.id_ == id_)
		if index < 0:
			return False

		self._change_list.add(id_)
		return True

	# Interface
	def getDictData(self) -> Dict:
		log_data_list: List[Dict] = []
		for log_data in self.log_data_list:
			log_data_list.append(log_data.getDictData())

		return {
			Control_Data.Label.LOG_DATA_LIST:	log_data_list,
			Control_Data.Label.INDEX:			self.index
		}

	def setDictData(self, data: Dict) -> None:
		# log data list
		log_data_list: List[Any] = \
			self._getDataFromDict_(data, str(Control_Data.Label.LOG_DATA_LIST), [])

		for item in log_data_list:
			log_data = Log_Data()
			log_data.setDictData(item)
			self.log_data_list.append(log_data)

		# index
		self.index = self._getDataFromDict_(data, str(Control_Data.Label.INDEX), 1)

	# Protected
	def _findIndex_(self, list_: List[Any], cmp: Callable[[Any], bool]) -> int:
		for index, item in enumerate(list_):
			if not cmp(item):
				continue
			return index
		return -1
