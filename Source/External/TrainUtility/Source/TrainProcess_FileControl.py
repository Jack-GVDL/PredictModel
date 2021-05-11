import os
from typing import *
from .TrainProcess import TrainProcess
from .ModelInfo import ModelInfo
from .FileControl import FileControl_Local


class TrainProcess_FileControlBuilder(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name 				= "FileControlBuilder"
		self.path_base:		str = ""  # this should exist
		self.path_folder:	str = ""  # this should not exist
		self.path_src:		str = ""

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	def setTargetPath(self, base: str, folder: str) -> None:
		self.path_base		= base
		self.path_folder	= folder

	# operation
	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		# create the base directory
		self.path_src = os.path.join(self.path_base, self.path_folder)
		if not os.path.isdir(self.path_src):
			os.mkdir(self.path_src)

		# create file control
		control = FileControl_Local()
		control.setLocalRoot(self.path_src)
		control.start()

		info.file_control = control

	# info
	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	# Protected
	def _getContent_(self, info: ModelInfo) -> str:
		return "Operation: " + self.name


class TrainProcess_FileControlUpdater(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name = "FileControlUpdater"

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# operation
	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		info.file_control.update()

	# Operation
	# ...

	# Protected
	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	# Protected
	def _getContent_(self, info: ModelInfo) -> str:
		return "Operation: " + self.name

