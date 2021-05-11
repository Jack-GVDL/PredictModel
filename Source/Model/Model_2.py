"""
Conv-LSTM
"""


from typing import *
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from ..External.TrainUtility import *
from .Model_Base import Model_Base
from .Model_ConvLSTM import ConvLSTM


class Model_2(
	Model_Base,
	Interface_CodePath):

	class Layer:
		FEATURE_EXTRACTOR:	int = 0
		CONV_LSTM:			int = 1
		AVG_POOL:			int = 2
		OUTPUT_FC:			int = 3
		SIZE_MAX:			int = 4

	def __init__(self):
		super().__init__()

		# ----- data -----
		# module
		self.feature_extractor = nn.Sequential(
			nn.Conv2d(4, 64, 3),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, 3),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

		self.lstm = ConvLSTM(input_dim=128, hidden_dim=64, kernel_size=[(3, 3)], num_layers=1, batch_first=True)

		self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
		self.output_fc = nn.Linear(1024, 3, bias=True)

		# layer
		self.layer = [None for _ in range(self.Layer.SIZE_MAX)]
		self.layer[self.Layer.FEATURE_EXTRACTOR]	= self.feature_extractor
		self.layer[self.Layer.CONV_LSTM]			= self.lstm
		self.layer[self.Layer.AVG_POOL]				= self.avgpool
		self.layer[self.Layer.OUTPUT_FC]			= self.output_fc

		# ----- operation -----
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def forward(self, input_tensor: torch.Tensor):
		# input_tensor: [B, T, C, H, W]
		x = input_tensor

		# down-sampler
		slice_list: List[Any] = []
		for i in range(x.shape[1]):
			time_slice = x[:, i, :, :, :]
			time_slice = self.feature_extractor(time_slice)
			slice_list.append(time_slice)

		x = torch.stack(slice_list, dim=1)

		# lstm
		x, _ = self.lstm(x)
		x = x[0]
		x = x[:, -1, :, :, :]  # src x: [B, T, C, H, W]

		# output fc
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.output_fc(x)

		return x

	# Interface
	def getCodePath(self) -> str:
		return __file__

	# Protected
	# ...
