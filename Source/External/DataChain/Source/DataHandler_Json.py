from typing import *
from .DataList import DataKey
from .DataList import DataList
from .DataFile import File_Json
from .DataHandler import DataHandler


class DataHandler_Json(DataHandler):

	def __init__(self):
		super().__init__()

		# data
		self.file_json = File_Json()

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# ...

	# Protected
	def _load_(self, data_list: DataList) -> bool:
		if not self.file_json.load():
			return False

		temp: Dict = self.file_json.data["DataList"]
		self._loadData_(temp, data_list)

		return True

	def _dump_(self, data_list: DataList) -> bool:
		temp: Dict = {}
		self._dumpData_(temp, data_list)
		data_dict: Dict = {
			"DataList": temp
		}

		self.file_json.data = data_dict
		if not self.file_json.dump():
			return False

		return True

	# TODO: not yet completed
	def _loadData_(self, data_dict: Dict, data_list: DataList) -> None:
		pass

	# TODO: not yet completed
	def _dumpData_(self, data_dict: Dict, data_list: DataList) -> None:
		pass
