from typing import *
from ..External.TrainUtility import *
import torch
import torch.nn as nn


class Model_Base(
	nn.Module,
	Interface_CodePath):

	def __init__(self):
		super().__init__()

		# data
		# ...

		# operation
		self.layer: 		List[nn.Module] 	= []
		self.child_list: 	List[Any] 			= []

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def forward(self, x: torch.Tensor) -> Any:
		raise NotImplementedError

	def addChild(self, child: Any) -> bool:
		index: int = self.child_list.index(child)
		if index >= 0:
			return False

		self.child_list.append(child)
		return True

	def rmChild(self, child: Any) -> bool:
		index: int = self.child_list.index(child)
		if index < 0:
			return False

		self.child_list.pop(index)
		return True

	def setGradient(self, layer: int, is_require_gradient: bool) -> bool:
		# check layer index
		# size of self.layer should be fixed after __init__
		if layer < 0 or layer >= len(self.layer):
			return False

		# get layer
		target_layer = self.layer[layer]

		# check if layer is mounted or not
		if target_layer is None:
			return False

		# set requires_grad
		# for each parameter, set if it require gradient or not
		for param in target_layer.parameters():
			param.requires_grad = is_require_gradient

		return True

	# Protected
	def getPathList(self) -> List[str]:
		result: List[str] = []
		for child in self.child_list:
			result.extend(child.getPathList())

		result.append(self.getCodePath())
		return result
