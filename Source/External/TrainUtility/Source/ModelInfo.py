from typing import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .Util_Interface import Interface_DictData
from .TrainProcess import TrainProcessControl
from .FileControl import FileControl_Base


# Data Structure
# basic
class ConfusionMatrix:

	def __init__(self, confusion_matrix: np.ndarray):
		super().__init__()

		# data
		# confusion matrix
		# row: actual / ground truth
		# col: prediction
		# confusion_matrix[row][col]
		self.matrix: np.ndarray = confusion_matrix

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	# get size of class, i.e. m
	def getSizeClass(self) -> int:
		# assumed: row == col
		return self.matrix.shape[0]

	# (TP + TN) / (TP + FP + FN + TN)
	def getAccuracy(self) -> float:
		assert self.matrix is not None

		# check if there is sample in the matrix
		# if no sample, just output 0
		size_sample = np.sum(self.matrix)
		if size_sample == 0:
			return 0.0

		# count TP and TN, i.e. item in the diagonal
		matrix_diagonal = np.diagonal(self.matrix)

		return np.sum(matrix_diagonal) / size_sample

	# TP / (TP + FP)
	def getPrecision(self, positive_list: np.ndarray) -> float:
		assert self.matrix is not None

		# first sort list
		# or may get wrong item in matrix[index][label]
		positive_list = np.sort(positive_list)

		# select row
		# then check if the resultant matrix is empty or not
		matrix = self.matrix[positive_list, :]
		if np.sum(matrix) == 0:
			return 0.0

		count_true = 0
		for index, label in enumerate(positive_list):
			count_true += matrix[index][label]

		return count_true / np.sum(matrix)

	# TP / (TP + FN)
	def getRecall(self, positive_list: np.ndarray) -> float:
		assert self.matrix is not None

		# first sort list
		# or may get wrong item in matrix[label][index]
		positive_list = np.sort(positive_list)

		# select col
		# then check if the resultant matrix is empty or not
		matrix = self.matrix[:, positive_list]
		if np.sum(matrix) == 0:
			return 0.0

		count_true = 0
		for index, label in enumerate(positive_list):
			count_true += matrix[label][index]

		return count_true / np.sum(matrix)

	# 2 * (Recall * Precision) / (Recall + Precision)
	def getF1Score(self, positive_list: np.ndarray) -> float:
		assert self.matrix is not None

		recall		= self.getRecall(positive_list)
		precision	= self.getPrecision(positive_list)

		if recall + precision == 0:
			return 0.0
		return 2 * (recall * precision) / (recall + precision)

	# single
	def getAccuracy_Single(self, row: int) -> float:
		assert self.matrix is not None

		# check if there is sample in the matrix
		# if no sample, just output 0
		size_sample = np.sum(self.matrix)
		if size_sample == 0:
			return 0.0

		# count TP and TN, i.e. item in the diagonal
		return self.matrix[row][row] / np.sum(self.matrix, axis=1)[row]

	def getPrecision_Single(self, class_: int) -> float:
		return self.getPrecision(np.array([class_]))

	def getRecall_Single(self, class_: int) -> float:
		return self.getRecall(np.array([class_]))

	def getF1Score_Single(self, class_: int) -> float:
		return self.getF1Score(np.array([class_]))

	# Operator Overload
	# TODO
	def __str__(self) -> str:
		raise NotImplementedError


class ModelInfo(Interface_DictData):

	class Stage:
		START:			int = 1
		END:			int = 2

		TRAIN:			int = 11
		VAL:			int = 12
		TEST:			int = 13

		TRAIN_START:	int = 21
		TRAIN_END:		int = 22
		VAL_START:		int = 23
		VAL_END:		int = 24
		TEST_START:		int = 25
		TEST_END:		int = 26

	def __init__(self):
		super().__init__()

		# ----- data -----
		# model
		self.model:			nn.Module	= None
		self.model_info:	Dict		= {}

		# train
		self.random_seed:	int		= 0
		self.epoch:			int		= 100
		self.batch_size: 	int		= 2
		self.iteration:		int 	= 0
		self.stage:			int		= 0

		self.optimizer 				= None
		self.scheduler				= None

		self.device_train			= None
		self.device_test			= None
		self.device_current			= None

		self.train_object:	Dict	= {}
		self.train_info: 	Dict	= {}  # other train parameter (e.g. momentum in SGD)

		# dataset
		self.dataset:			Dict	= {}
		self.dataset_info:		Dict	= {}
		self.dataset_object:	Dict	= {}

		# result
		self.result:	Dict		= {}
		self.log: 		List[str]	= []

		# process
		self.process_control: TrainProcessControl = TrainProcessControl()

		# file system
		self.file_control:	FileControl_Base	= None
		self.file_info: 	Dict				= {}

		# ops and hook
		# difference between hook and ops
		# ops: 	must
		# hook: optional
		self.ops_forwarding:				Callable[[Any, Any, Any], Any] 								= None
		self.ops_backwarding:				Callable[[Any, Any], Any] 									= None
		self.ops_get_dataloader:			Callable[[Any, Dataset], DataLoader] 						= None  # (ModelInfo, Dataset) -> DataLoader
		self.ops_pack_batch_result:			Callable[[Any, Any, Any, Any, Any], Any]					= None  # (ModelInfo, x, y, predict, loss) -> packed_result
		self.ops_pack_epoch_result:			Callable[[Any, Any], Any]									= None  # (ModelInfo, batch_result_list) -> packed_result
		self.ops_handle_train_result:		Callable[[Any, Any], None] 									= None  # (ModelInfo, train_result) -> None
		self.ops_handle_validate_result: 	Callable[[Any, Any], None] 									= None  # (ModelInfo, validate_result) -> None
		self.ops_handle_test_result:		Callable[[Any, Any], None]									= None  # (ModelInfo, test_result) -> None

		self.hook_update_model:				Callable[[Any], None] = None
		self.hook_update_training:			Callable[[Any], None] = None

		# ----- operation -----
		# ...

	def __del__(self):
		return

	# Operation
	def executeProcess(self, stage: int, data: Dict) -> None:
		self.process_control.execute(stage, self, data, self.log)

	def getDictData(self) -> Dict:
		return {
			# model
			"ModelInfo":		self.model_info,

			# train
			"RandomSeed":		self.random_seed,
			"Epoch":			self.epoch,
			"BatchSize":		self.batch_size,
			"TrainInfo":		self.train_info,

			# dataset
			"DatasetInfo":		self.dataset_info,

			# result
			"Result":			self.result,
			"Log":				self.log
		}

	def setDictData(self, data: Dict) -> None:
		# get from dict
		self.model_info 	= self._getDataFromDict_(data, "ModelInfo",		self.model_info)
		self.epoch 			= self._getDataFromDict_(data, "Epoch", 		self.epoch)
		self.batch_size		= self._getDataFromDict_(data, "BatchSize", 	self.batch_size)
		self.train_info 	= self._getDataFromDict_(data, "TrainInfo",		self.train_info)
		self.dataset_info	= self._getDataFromDict_(data, "DatasetInfo",	self.dataset_info)
		self.result			= self._getDataFromDict_(data, "Result",		self.result)
		self.log			= self._getDataFromDict_(data, "Log",			self.log)

		# update
		if self.hook_update_model is not None:
			self.hook_update_model(self)

		if self.hook_update_training is not None:
			self.hook_update_training(self)
