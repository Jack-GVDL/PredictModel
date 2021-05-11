from typing import *
import json
import os
from os.path import isfile, join
from .DataList import DataBase


class FileBase:

	def __init__(self):
		super().__init__()

		# data
		self.file_list:	List[str] 	= []
		self.folder: 	str	  		= ""
		self.data:		Any		  	= None

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	def searchFolder(self, filter_: Callable[[str], bool]) -> bool:
		# foreach file in the folder
		# find out the valid file
		#
		# list file in folder
		# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
		file_list: List[str] = [f for f in os.listdir(self.folder) if isfile(join(self.folder, f))]
		self.file_list = list(filter(filter_, file_list))
		return True

	# operation
	def load(self) -> bool:
		if not self.file_list:
			return True

		# foreach file in file_list
		for file in self.file_list:

			# operation to invalid file: ignore
			if file == "":
				continue
			self._load_(join(self.folder, file))

		return True

	def dump(self) -> bool:
		if not self.file_list:
			return True

		# foreach file in file_list
		for file in self.file_list:

			# operation to invalid file: ignore
			if file == "":
				continue
			self._dump_(join(self.folder, file))

		return True

	# Protected
	def _load_(self, file: str) -> bool:
		raise NotImplementedError

	def _dump_(self, file: str) -> bool:
		raise NotImplementedError


class File_Json(FileBase):

	def __init__(self):
		super().__init__()
		
		# data
		self.is_compact: bool = True
		
		# operation
		# ...
		
	def __del__(self):
		return
		
	# Operation
	def _load_(self, file: str) -> bool:
		with open(file, "r") as f:
			self.data = json.load(f)
		return True

	def _dump_(self, file: str) -> bool:
		indent: int = 2

		if self.is_compact:
			indent = None

		with open(file, "w") as f:
			json.dump(self.data, f, indent=indent, separators=(",", ":"))
		return True


class File_Text(FileBase):

	def __init__(self):
		super().__init__()
		
		# data
		self.data = []
		
		# operation
		# ...
		
	def __del__(self):
		return
		
	# Operation
	def _load_(self, file: str) -> bool:
		with open(file, "r") as f:
			for line in f:
				self.data.append(line)
		return True

	def _dump_(self, file: str) -> bool:
		with open(file, "w") as f:
			for line in self.data:
				f.writelines(line)
		return True
