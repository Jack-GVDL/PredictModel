from typing import *
import cmath
import datetime
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from ..External.DataChain import *
from ..External.Util_Cmd import *
from .Handler_DataUpdateLog import Handler_DataUpdateLog


# Data Structure
class Dataset_Base(Dataset):

	# Static Function
	# ...

	def __init__(self, data_list: DataList):
		super().__init__()

		# assert
		assert data_list is not None

		# data
		self._data_list:	DataList = data_list
		self._data_key:		DataKey	 = data_list.data_key

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	# ...

	# Protected
	def _computeSingleData_(self, data: DataBase) -> Any:
		raise NotImplementedError

	# Operator Overload
	def __getitem__(self, index: int) -> Any:
		# get data from data list
		data: DataBase = self._data_list[index]

		# compute data
		return self._computeSingleData_(data)

	def __len__(self) -> int:
		return len(self._data_list)

