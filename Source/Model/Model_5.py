from typing import *
import torch
import torch.nn as nn
from torchvision import models
from ..External.TrainUtility import *
from .Model_Base import Model_Base
from .Model_ConvLSTM_3d_2 import ConvLSTM3d_2


# Data Structure
class Model_5(Model_Base):

	class Layer:
		INPUT_EXTRACTOR:	int = 0
		CONV_LSTM:			int = 1
		AVG_POOL:			int = 2
		OUTPUT_EXTRACTOR:	int = 3
		OUTPUT_FC:			int = 4
		SIZE_MAX:			int = 5

	def __init__(self):
		super().__init__()

		# ----- data -----
		# config
		self.size_hidden_layer: int = 3

		# module
		self.input_extractor = nn.Sequential(
			nn.Conv2d(4, 64, 3),
			# nn.Dropout(inplace=False),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 64, 3),
			nn.Conv2d(64, 64, 3),
			# nn.Conv2d(64, 64, 3),
			# nn.Dropout(inplace=False),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

		# size of height level: 16, channel per height: 4
		# where format of tensor
		# [B, T, C, D, H, W]
		self.lstm = ConvLSTM3d_2(dim=4, kernel_size=(3, 3, 3))

		self.output_extractor = nn.Sequential(
			nn.Conv2d(	4, 64, 3),
			nn.ReLU(	inplace=False),
		)

		self.avgpool = nn.AdaptiveAvgPool2d((16, 16))
		self.output_fc = nn.Linear(1024, 3, bias=True)

		# layer
		self.layer = [None for _ in range(self.Layer.SIZE_MAX)]

		self.layer[self.Layer.INPUT_EXTRACTOR]	= self.input_extractor
		self.layer[self.Layer.CONV_LSTM]		= self.lstm
		self.layer[self.Layer.OUTPUT_EXTRACTOR]	= self.output_extractor
		self.layer[self.Layer.AVG_POOL]			= self.avgpool
		self.layer[self.Layer.OUTPUT_FC]		= self.output_fc

		# ----- operation -----
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def forward(self, input_tensor: torch.Tensor) -> Any:
		# input tensor: [B, T, C, H, W]
		x = input_tensor

		size_height: 	int = 16
		size_channel: 	int = 4

		# ----- input extractor -----
		slice_list: List[Any] = []
		for i in range(x.shape[1]):
			time_slice = x[:, i, :, :, :]
			time_slice = self.input_extractor(time_slice)
			slice_list.append(time_slice)

		# [[B, C, H, W], ...] -> [B, T, C, H, W]
		x = torch.stack(slice_list, dim=1)

		# ----- 2D -> 3D -----
		# number of height level = 16
		height_list: List[Any] = []
		for i in range(size_height):
			height_slice = x[:, :, (i * size_channel):((i + 1) * size_channel), :, :]
			height_slice = height_slice.reshape((
				height_slice.shape[0],	# B
				height_slice.shape[1],	# T
				height_slice.shape[2],	# C
				1,						# D
				height_slice.shape[3],	# H
				height_slice.shape[4]))	# w

			height_list.append(height_slice)

		x = torch.cat(height_list, dim=3)

		# ----- LSTM -----
		# compute mask
		t 			= x.shape[1]
		mask_list 	= []

		for _ in range(t):
			mask_list.append(True)
			for _ in range(self.size_hidden_layer):
				mask_list.append(False)

		# lstm
		sequence, _ = self.lstm(x, mask_list)  # output_list, state_list = self.lstm(x)
		x 			= sequence[-1]  # x: format: [B, C, D, H, W]
		sequence	= torch.stack(sequence, dim=1)

		# ----- output fc -----
		# split height level
		result_list: List[Any] = []
		for i in range(size_height):
			result = x[:, :, i, :, :]

			# result = self.output_extractor(result)
			result = self.avgpool(result)

			result = torch.flatten(result, 1)
			result = self.output_fc(result)

			result = result.reshape((result.shape[0], 1, result.shape[1]))
			result_list.append(result)  # result: [B, 1, 3]

		# RET
		# format: [B, D, 3]
		x = torch.cat(result_list, dim=1)

		return [x, sequence]

	# Protected
	def getCodePath(self) -> str:
		return __file__
