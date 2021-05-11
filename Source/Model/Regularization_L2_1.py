from typing import *
from .Model_Base import Model_Base
import torch


# Data Structure
class Regularization_L2_1(Model_Base):

	def __init__(self, model: Model_Base, weight_decay, p=2, name=""):
		super().__init__()

		# data
		self.model			= model
		self.weight_decay	= weight_decay
		self.p				= p
		self.name			= name

		self.weight_list	= self._getWeightList_(model)

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def forward(self, model) -> Any:
		self.weight_list = self._getWeightList_(model)
		loss = self._getLoss_(self.weight_list, self.weight_decay, p=self.p)
		return loss

	# Protected
	def _getWeightList_(self, model) -> List[Any]:
		weight_list = []

		for name, param in model.named_parameters():
			if not param.requires_grad:
				continue
			if self.name not in name:
				continue
			weight = (name, param)
			weight_list.append(weight)

		return weight_list

	def _getLoss_(self, weight_list, weight_decay, p):
		loss = .0

		for name, weight in weight_list:
			l2_regularization = torch.norm(weight, p=p)
			loss += l2_regularization

		loss *= weight_decay
		return loss
