from typing import *
from .Util_Interface import Interface_CodePath
from .TrainProcess import TrainProcess
from .ModelInfo import ModelInfo
from .FileControl_FileNode import FileNode_Transfer


class TrainProcess_PythonFile(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name = "PythonFile"

		self._save_list: List[Tuple[Interface_CodePath, str]] = []

		# operation
		# ...

	def __del__(self):
		return

	# Property
	@property
	def save_list(self):
		return self._save_list.copy()

	# Operation
	# data
	def setData(self, data: Dict) -> None:
		self._save_list = self._getDataFromDict_(data, "save_list", self._save_list)

	def getData(self) -> Dict:
		return {
			"save_list": self._save_list
		}

	# operation
	def addPythonFile(self, obj: Interface_CodePath, filename: str) -> bool:
		self._save_list.append((obj, filename))
		return True

	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		for data in self._save_list:
			obj			= data[0]
			filename	= data[1]

			node = FileNode_Transfer(obj.getCodePath())
			node.name 		= filename
			node.extension	= "py"

			info.file_control.mountFile(".", node)

	# info
	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getInfo(self) -> List[List[str]]:
		info: List[List[str]] = []

		# ----- save list -----
		save_list = map(lambda x: ["", x[1]], self._save_list)
		save_list = list(save_list)

		# if the save_list is not empty
		# then the first item will be assigned with a parameter name (save_list)
		if save_list:
			save_list[0][0] = "save_list"

		info.extend(save_list)

		return info

	# Protected
	def _getContent_(self, info: ModelInfo) -> str:
		result: str = ""
		result 		+= "Operation: save code file\n"
		result		+= "File:\n"

		for data in self._save_list:
			obj		= data[0]
			result 	+= obj.getCodePath() + "\n"

		return result
