from typing import *
import torch
import torch.nn as nn
from torchvision import models
from ..External.TrainUtility import *
from .Model_Base import Model_Base
from .Model_ConvLSTM_2d_2 import ConvLSTM2d_2
from .Model_ConvLSTM_3d_2 import ConvLSTM3d_2


# Data Structure
class Model_7(Model_Base):

	class Layer:
		INPUT_LSTM:		int = 0
		CONV_LSTM:		int = 1
		AVG_POOL:		int = 2
		OUTPUT_FC:		int = 3
		SIZE_MAX:		int = 4

	def __init__(self):
		super().__init__()

		# ----- data -----
		# config
		self.size_hidden_layer: int = 1

		# module
		self.input_lstm = ConvLSTM2d_2(
			dim=4,
			kernel_size=(3, 3),
			size_time=16
		)

		# size of height level: 16, channel per height: 4
		# where format of tensor
		# [B, T, C, D, H, W]
		self.conv_lstm = ConvLSTM3d_2(dim=4, kernel_size=(3, 3, 3))

		self.avgpool = nn.AdaptiveAvgPool2d((16, 16))
		self.output_fc = nn.Linear(1024, 3, bias=True)

		# layer
		self.layer = [None for _ in range(self.Layer.SIZE_MAX)]

		self.layer[self.Layer.INPUT_LSTM]		= self.input_lstm
		self.layer[self.Layer.CONV_LSTM]		= self.conv_lstm
		self.layer[self.Layer.AVG_POOL]			= self.avgpool
		self.layer[self.Layer.OUTPUT_FC]		= self.output_fc

		# ----- operation -----
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def forward(self, input_data: Any) -> Any:
		# x: 			[B, T, C, H, W]
		# time_mask: 	[B, T]
		x 			= input_data[0]
		time_mask	= input_data[1].cpu().tolist()

		size_height: 	int = 16
		size_channel: 	int = 4

		# ----- input extractor -----
		t = x.shape[1]

		slice_list: List[Any] = []
		for i in range(t):
			time_slice 		= x[:, i, :, :, :]
			time_slice, _ 	= self.input_lstm(time_slice)
			time_slice 		= torch.stack(time_slice, dim=2)  # dst: [B, C, D, H, W]
			slice_list.append(time_slice)

		x = torch.stack(slice_list, dim=1)  # dst: [B, T, C, D, H, W]

		# ----- LSTM -----
		# compute mask
		size_time 	= len(time_mask)
		mask_list	= []

		for t in range(size_time):
			mask_list.append(time_mask[t])
			for _ in range(self.size_hidden_layer):
				mask_list.append(False)

		# lstm
		sequence, _ = self.conv_lstm(x, mask_list)  # output_list, state_list = self.lstm(x)
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
