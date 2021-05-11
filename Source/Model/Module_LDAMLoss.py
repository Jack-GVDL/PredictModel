import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..External.TrainUtility import *


class LDAMLoss(
	nn.Module,
	Interface_CodePath):

	def __init__(self, class_size_list: torch.Tensor, device, max_m=0.5, weight=None, s=30):
		super(LDAMLoss, self).__init__()

		assert s > 0

		# data
		class_size_list = class_size_list.cpu().numpy()

		m_list 		= 1.0 / np.sqrt(np.sqrt(class_size_list))  # cls_num_list ^ (1 / 4)
		m_list 		= m_list * (max_m / np.max(m_list))

		# m_list 		= torch.cuda.FloatTensor(m_list)
		m_list		= torch.tensor(m_list, dtype=torch.float).to(device)

		self.m_list = m_list
		self.s 		= s
		self.weight = weight

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def forward(self, x, target):
		index = torch.zeros_like(x, dtype=torch.uint8)
		index.scatter_(1, target.data.view(-1, 1), 1)

		index_float = index.type(torch.cuda.FloatTensor)
		batch_m 	= torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
		batch_m 	= batch_m.view((-1, 1))
		x_m 		= x - batch_m

		output = torch.where(index, x_m, x)
		return F.cross_entropy(self.s * output, target, weight=self.weight)

	# Interface
	def getCodePath(self) -> str:
		return __file__

	# Protected
	# ...
