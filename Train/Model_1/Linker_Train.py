from typing import *
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from Source import *


# Data Structure
class File_Linker_Train(Interface_CodePath):

	# Interface
	def getCodePath(self) -> str:
		return __file__


class TrainProcess_Scheduler(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name = "Scheduler"

		# operation
		# only update the optimizer before the iteration start
		self.addStage(ModelInfo.Stage.TRAIN_START)

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		# iteration 0 - 100
		if info.iteration == 0:
			self._updateOptimizer_1_(info)

		# iteration: 100 - 200
		elif info.iteration == 100:
			self._updateOptimizer_2_(info)

		# iteration: after 200
		elif info.iteration == 200:
			self._updateOptimizer_3_(info)

	def _updateOptimizer_1_(self, info: ModelInfo) -> None:
		# ----- set require_grad -----
		model: Model_1 = info.model

		model.setGradient(model.Layer.INPUT_CONV,			True)
		model.setGradient(model.Layer.FEATURE_EXTRACTOR,	False)
		model.setGradient(model.Layer.OUTPUT_FC,			True)

		# only update fc in feature_extractor
		for param in model.feature_extractor.fc.parameters():
			param.requires_grad = True

		# ----- update optimizer -----
		# get parameter that needed to update
		# then save it to the newly created optimizer
		parameter_list = self._getParameterList_RequireGrad_(model)
		info.optimizer = optim.SGD(
			parameter_list,
			lr=1e-4,
			momentum=0.9,
			weight_decay=5e-4,
			nesterov=True)
		info.scheduler = lr_scheduler.StepLR(info.optimizer, step_size=80, gamma=0.2)

	def _updateOptimizer_2_(self, info: ModelInfo) -> None:
		# ----- set require_grad -----
		model: Model_1 = info.model

		model.setGradient(model.Layer.INPUT_CONV,			True)
		model.setGradient(model.Layer.FEATURE_EXTRACTOR,	False)
		model.setGradient(model.Layer.OUTPUT_FC,			True)

		# only update layer4 in feature_extractor
		for param in model.feature_extractor.layer4.parameters():
			param.requires_grad = True

		# ----- update optimizer -----
		# get parameter that needed to update
		# then save it to the newly created optimizer
		parameter_list = self._getParameterList_RequireGrad_(info.model)
		info.optimizer = optim.SGD(
			parameter_list,
			lr=8e-4,
			momentum=0.9,
			weight_decay=5e-4,
			nesterov=True)
		info.scheduler = lr_scheduler.StepLR(info.optimizer, step_size=80, gamma=0.2)

	def _updateOptimizer_3_(self, info: ModelInfo) -> None:
		# ----- set require_grad -----
		model: Model_1 = info.model

		model.setGradient(model.Layer.INPUT_CONV,			True)
		model.setGradient(model.Layer.FEATURE_EXTRACTOR,	False)
		model.setGradient(model.Layer.OUTPUT_FC,			True)

		# only update fc in feature_extractor
		for param in model.feature_extractor.fc.parameters():
			param.requires_grad = True

		# ----- update optimizer -----
		# get parameter that needed to update
		# then save it to the newly created optimizer
		parameter_list = self._getParameterList_RequireGrad_(info.model)
		info.optimizer = optim.SGD(
			parameter_list,
			lr=2e-5,
			momentum=0.9,
			weight_decay=2e-4,
			nesterov=True)
		info.scheduler = lr_scheduler.StepLR(info.optimizer, step_size=80, gamma=0.2)

	def _getParameterList_RequireGrad_(self, model: nn.Module) -> List[Any]:
		# get parameter that needed to update
		parameter_list = []

		for _, param in model.named_parameters():
			if not param.requires_grad:
				continue
			parameter_list.append(param)

		return parameter_list


class TrainProcess_Result(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name = "ResultData"

		self.file_model_info 			= FileNode_PlainText(None)
		self.file_model_info.name		= "ModelInfo"
		self.file_model_info.extension	= "json"

		self.file_state_dict 			= FileNode_StateDict(None)
		self.file_state_dict.name 		= "StateDict"
		self.file_state_dict.extension 	= "tar"

		self.best_state_dict:	Any		= None
		self.best_accuracy: 	float 	= .0
		self.best_loss:			float	= float("inf")
		self.best_iteration:	int 	= -1

		# operation
		self.addStage(ModelInfo.Stage.START)
		self.addStage(ModelInfo.Stage.TRAIN_END)
		self.addStage(ModelInfo.Stage.VAL_END)
		self.addStage(ModelInfo.Stage.TEST_START)
		self.addStage(ModelInfo.Stage.TEST_END)
		self.addStage(ModelInfo.Stage.END)

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def setStateDict(self, state_dict: Any) -> None:
		self.file_state_dict.setStateDict(state_dict)

	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		# start
		if stage == ModelInfo.Stage.START:

			info.file_control.mountFile(".", self.file_state_dict)
			info.file_control.mountFile(".", self.file_model_info)
			return

		# train end
		if stage == ModelInfo.Stage.TRAIN_END:
			return

		# validation end
		if stage == ModelInfo.Stage.VAL_END:

			# get loss and save best state_dict
			loss = info.result["Val_Loss"][-1]
			if loss < self.best_loss:

				self.best_loss 		= loss
				self.best_iteration = info.iteration

				# save to info
				info.result["Best_Loss"] 		= self.best_loss
				info.result["Best_Iteration"] 	= self.best_iteration

				# save state dict
				self.best_state_dict = info.model.state_dict()
				self.file_state_dict.setStateDict(self.best_state_dict)

			# save model info
			data = info.getDictData()
			data = json.dumps(data, indent=2)
			self.file_model_info.setData(data)
			return

		# test start
		if stage == ModelInfo.Stage.TEST_START:
			info.model.load_state_dict(self.best_state_dict)
			return

		# test end
		if stage == ModelInfo.Stage.TEST_END:
			return

		# end
		if stage == ModelInfo.Stage.END:

			# save model info
			data = info.getDictData()
			data = json.dumps(data, indent=2)
			self.file_model_info.setData(data)
			return

	# Protected
	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(stage, info)

	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(stage, info)

	def _getContent_(self, stage: int, info: ModelInfo) -> str:
		if stage == ModelInfo.Stage.START:
			return self._getContent_Start_(info)

		if stage == ModelInfo.Stage.TRAIN_END:
			return self._getContent_TrainEnd_(info)

		if stage == ModelInfo.Stage.VAL_END:
			return self._getContent_ValidationEnd_(info)

		if stage == ModelInfo.Stage.TEST_END:
			return self._getContent_TestEnd_(info)

		return ""

	def _getContent_Start_(self, info: ModelInfo) -> str:
		return ""

	def _getContent_TrainEnd_(self, info: ModelInfo) -> str:
		content: str = ""

		# stage
		content += f"[Train]: "

		# iteration
		content += f"Iteration: {info.iteration}"
		content += f"; "

		# loss
		loss = info.result["Train_Loss"][-1]
		content += f"Loss: {loss:.5f}"

		return content

	def _getContent_ValidationEnd_(self, info: ModelInfo) -> str:
		content: str = ""

		# stage
		content += f"[Val]: "

		# iteration
		content += f"Iteration: {info.iteration}"
		content += f"; "

		# loss
		loss = info.result["Val_Loss"][-1]
		content += f"Loss: {loss:.5f}"
		content += "; "

		# best score
		content += f"Best: "
		content += f"loss: {self.best_loss:.5f}, "
		content += f"iteration: {self.best_iteration}"

		return content

	def _getContent_TestEnd_(self, info: ModelInfo) -> str:
		content: str = ""

		# stage
		content += f"[Test]: "

		# loss
		loss = info.result["Test_Loss"][-1]
		content += f"Loss: {loss:.5f}"

		return content


class Linker_Train:

	def __init__(self):
		super().__init__()

		# data
		self.process_scheduler	= TrainProcess_Scheduler()
		self.process_result		= TrainProcess_Result()

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	# ops
	def batchData(self, data_list: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.tensor, torch.tensor]:
		# assert: data_list is not empty
		assert data_list

		# to list
		x_list: List[torch.Tensor]	= []
		y_list: List[int]			= []

		for data in data_list:
			x_list.append(data[0])
			y_list.append(data[1])

		# to tensor, ndarray
		result_x = torch.cat(x_list)
		result_y = torch.tensor(y_list)

		return result_x, result_y

	def Ops_getLoss(self, info: ModelInfo, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		# ----- cross entropy -----
		# if info.stage == ModelInfo.Stage.VAL or \
		# 	info.stage == ModelInfo.Stage.TEST:
		# 	return self._getLoss_CrossEntropy_(info, predict, target)

		# ----- LSTM -----
		# get ratio
		if info.stage == ModelInfo.Stage.TRAIN:
			class_size_list = info.dataset_object["Ratio_Train"]
		elif info.stage == ModelInfo.Stage.VAL:
			class_size_list = info.dataset_object["Ratio_Val"]
		else:
			class_size_list = info.dataset_object["Ratio_Test"]

		# TODO: currently the ratio is fixed
		class_weight_list	= torch.tensor([1, 1, 1], 		dtype=torch.float)
		class_size_list		= torch.tensor(class_size_list, dtype=torch.float)

		# get loss
		return self._getLoss_LDAM_(info, predict, target, class_size_list, class_weight_list)

	def Ops_getDataLoader(self, info: ModelInfo, dataset: Dataset_Base) -> DataLoader:
		# validation
		if info.stage == ModelInfo.Stage.VAL:
			return DataLoader(dataset, shuffle=True, batch_size=1, collate_fn=self.batchData)

		# training
		return DataLoader(dataset, shuffle=True, batch_size=info.batch_size, collate_fn=self.batchData)

	def Ops_packBatchResult(
			self, info: ModelInfo, x: torch.Tensor, y: torch.Tensor, predict: torch.Tensor, loss: torch.Tensor) -> Any:

		# to numpy or integer
		y 		= y.detach().cpu().numpy()
		predict = predict.detach().cpu().numpy()
		loss	= loss.detach().cpu().item()

		# argmax
		predict = np.argmax(predict, axis=1)

		# compute confusion matrix
		matrix = self._computeConfusionMatrix_(predict, y, 3)

		return [matrix, loss]

	def Ops_packEpochResult(self, info: ModelInfo, result_list: List[Any]) -> Any:
		matrix 	= np.zeros((3, 3), dtype=np.int)
		loss 	= 0

		for result in result_list:
			matrix += result[0]
			loss += result[1]

		loss /= len(result_list)

		return [matrix, loss]

	def Ops_handleTrainResult(self, info: ModelInfo, result: np.ndarray) -> None:
		# confusion matrix
		if "Train_Matrix" not in info.result.keys():
			info.result["Train_Matrix"] = []
		info.result["Train_Matrix"].append(result[0].tolist())

		# loss
		if "Train_Loss" not in info.result.keys():
			info.result["Train_Loss"] = []
		info.result["Train_Loss"].append(result[1])

	def Ops_handleValidateResult(self, info: ModelInfo, result: np.ndarray) -> None:
		# confusion matrix
		if "Val_Matrix" not in info.result.keys():
			info.result["Val_Matrix"] = []
		info.result["Val_Matrix"].append(result[0].tolist())

		# loss
		if "Val_Loss" not in info.result.keys():
			info.result["Val_Loss"] = []
		info.result["Val_Loss"].append(result[1])

	def Ops_handleTestResult(self, info: ModelInfo, result: np.ndarray) -> None:
		# confusion matrix
		if "Test_Matrix" not in info.result.keys():
			info.result["Test_Matrix"] = []
		info.result["Test_Matrix"].append(result[0].tolist())

		# loss
		if "Test_Loss" not in info.result.keys():
			info.result["Test_Loss"] = []
		info.result["Test_Loss"].append(result[1])

	# Protected
	def _computeConfusionMatrix_(self, predict: np.ndarray, y: np.ndarray, size_class: int) -> np.ndarray:
		"""
		Compute the confusion matrix
		input should be in shape of [N], where N is the number of sample
		the value should within [0, ... n - 1], where n is the number of class

		:param predict:		[N] x-class (from model)
		:param y:			[N] y-class (ground truth)
		:param size_class:	size of class: n
		:return:			confusion_matrix[ground_truth][predicted]
		"""
		# it is assumed that the size of predict_class and y_class are the same
		n = size_class

		# confusion matrix is in shape of [n, n]
		# confusion_matrix: List[List[int]] = [[0 for x in range(n)] for y in range(n)]
		matrix = np.zeros((n, n), dtype=np.int32)

		# TODO: find a way to do the parallel processing
		# foreach sample
		for i in range(y.shape[0]):
			row = y[i]
			col = predict[i]
			matrix[row][col] += 1

		return matrix

	def _getLoss_LDAM_(
		self,
		info: ModelInfo, predict: torch.Tensor, target: torch.Tensor,
		class_size_list: torch.Tensor, class_weight_list: torch.Tensor
	) -> torch.Tensor:

		# class_weight_list	= torch.FloatTensor(class_weight_list).to(info.device_current)
		# class_size_list		= torch.FloatTensor(class_size_list).to(info.device_current)

		class_weight_list	= class_weight_list.to(info.device_current)
		class_size_list		= class_size_list.to(info.device_current)

		loss_ldam = LDAMLoss(class_size_list, info.device_current, weight=class_weight_list).to(info.device_current)

		loss = loss_ldam(predict, target)
		return loss

	def _getLoss_CrossEntropy_(self, info: ModelInfo, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		loss = F.cross_entropy(predict, target).to(info.device_current)
		return loss
