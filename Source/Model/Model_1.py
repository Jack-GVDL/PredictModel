"""
Reference:
HeightEstimation: model
"""


from typing import *
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from ..External.TrainUtility import *
from .Model_Base import Model_Base


class Model_1(
	Model_Base,
	Interface_CodePath):

	class Layer:
		INPUT_CONV:			int = 0
		FEATURE_EXTRACTOR:	int = 1
		OUTPUT_FC:			int = 2
		SIZE_MAX:			int = 3

	def __init__(self):
		super().__init__()

		# ----- data -----
		# module
		self.input_conv 			= nn.Conv2d(8, 3, 1)

		self.feature_extractor 		= models.resnet50(pretrained=True)
		size_feature 				= self.feature_extractor.fc.in_features
		self.feature_extractor.fc 	= nn.Identity()

		self.output_fc 				= nn.Linear(size_feature, 3, bias=False)

		# layer
		self.layer = [None for _ in range(self.Layer.SIZE_MAX)]
		self.layer[self.Layer.INPUT_CONV] 			= self.input_conv
		self.layer[self.Layer.FEATURE_EXTRACTOR]	= self.feature_extractor
		self.layer[self.Layer.OUTPUT_FC] 			= self.output_fc

		# child list
		# ...

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
		x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])

		x = self.input_conv(x)
		x = self.feature_extractor(x)
		x = self.output_fc(x)

		return x

	# Interface
	def getCodePath(self) -> str:
		return __file__

	# Protected
	# ...
