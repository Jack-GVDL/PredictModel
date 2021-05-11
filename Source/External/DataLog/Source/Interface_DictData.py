from typing import *


# Data Structure
class Interface_DictData:

	def __init__(self) -> None:
		super().__init__()
		
		# data
		# ...

		# operation
		# ...

	def __del__(self) -> None:
		return

	# Operation
	def getDictData(self) -> Dict:
		return {}

	def setDictData(self, data: Dict) -> None:
		pass

	# Protected
	def _getDataFromDict_(self, dict_: Dict, key: str, default_value: Any):
		if key not in dict_:
			return default_value
		return dict_[key]
