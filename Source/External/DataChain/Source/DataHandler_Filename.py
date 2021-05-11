from typing import *
from .DataList import DataBase
from .DataList import DataList
from .DataFile import FileBase
from .DataHandler import DataHandler


class DataHandler_Filename(DataHandler):

	def __init__(self):
		super().__init__()

		# data
		self.file = FileBase()

		# only can load from filename but cannot do dumping
		self.func_load:	Callable[[List[Any], str], bool] = None

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	# ...

	# Protected
	def _load_(self, data_list: DataList) -> bool:
		# check
		if self.func_load is None:
			return False

		# foreach filename
		for filename in self.file.file_list:
			temp: List[Any] = []
			if not self.func_load(temp, filename):
				continue

			data: DataBase = data_list.createData()
			data.setContent(temp, is_inplace=True)

		return True

	def _dump_(self, data_list: DataList) -> bool:
		raise NotImplementedError
