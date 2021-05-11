from typing import *
from .DataList import DataBase
from .DataList import DataList
from .DataFile import File_Text
from .DataHandler import DataHandler


class DataHandler_Text(DataHandler):

	def __init__(self):
		super().__init__()

		# data
		self.file_text 		= File_Text()
		self.separator:	str = "\n"

		self.func_load:	Callable[[List[Any], str], None]	= None
		self.func_dump:	Callable[[List[Any]], str]			= None

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# ...

	# Protected
	def _load_(self, data_list: DataList) -> bool:
		# check
		if self.func_load is None:
			return False

		# load from text file
		if not self.file_text.load():
			return False
		content_list: List[str] = self.file_text.data

		# foreach content, where content is not yet separated by separator
		for content in content_list:
			item_list = content.split(self.separator)

			# foreach separated item
			# ignore empty item
			for item in item_list:
				if not item:
					continue

				# loading may be failed
				# first save the data to a buffer before loading is confirmed to be a success
				temp: List[Any] = []
				if not self.func_load(temp, content):
					continue

				# load is success, allocate space for data
				data: DataBase = data_list.createData()
				data.setContent(temp, is_inplace=True)

		return True

	def _dump_(self, data_list: DataList) -> bool:
		# check
		if self.func_dump is None:
			return False

		# get content from DataList
		content_list: List[str] = []
		for data in data_list:
			content:	str = ""
			content 		+= self.func_dump(data.data)
			content 		+= self.separator
			content_list.append(content)

		# dump to text file
		self.file_text.data = content_list
		if not self.file_text.dump():
			return False

		return True
