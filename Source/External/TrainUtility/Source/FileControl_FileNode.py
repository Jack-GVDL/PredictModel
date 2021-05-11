from typing import *
import os
import shutil
import pathlib
import torch
from .FileControl import FileNode_Base


# Data Structure
class FileNode_Transfer(FileNode_Base):

	def __init__(self, path_src: str):
		super().__init__()

		# data
		self.path_src: str = path_src

		# operation
		# marked as require_dump once the path_src is provided
		self._is_require_dump = True

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def _dump_(self, path: str, filename: str) -> bool:
		shutil.copyfile(self.path_src, os.path.join(path, filename))
		return True

	# Protected
	# ...


class FileNode_PlainText(FileNode_Base):

	def __init__(self, data: Any):
		super().__init__()

		# data
		self.data = data

		# operation
		# marked as require_dump once the path_src is provided
		self._is_require_dump = True

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def setData(self, data: Any) -> None:
		self.data = data
		self.requestDump()

	def _dump_(self, path: str, filename: str) -> bool:
		if self.data is None:
			return False

		with open(os.path.join(path, filename), "w") as f:
			f.write(self.data)
		return True

	# Protected
	# ...


class FileNode_StateDict(FileNode_Base):

	def __init__(self, state_dict):
		super().__init__()

		# data
		self.state_dict = state_dict

		# operation
		# marked as require_dump once the path_src is provided
		if self.state_dict is not None:
			self._is_require_dump = True

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	# if the state_dict is updated, then the file is required to be updated
	def setStateDict(self, state_dict: Any) -> None:
		self.state_dict = state_dict
		self.requestDump()

	def _dump_(self, path: str, filename: str) -> bool:
		if self.state_dict is None:
			return False

		torch.save(self.state_dict, os.path.join(path, filename))
		return True

	# Protected
	# ...
