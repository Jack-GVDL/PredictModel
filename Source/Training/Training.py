from typing import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm
from ..External.TrainUtility import *


# Function
def trainBatch(info: ModelInfo, x, y, is_val=False) -> Any:
	# forwarding
	predict, loss = info.ops_forwarding(info, x, y)

	# invalid loss
	if torch.isnan(loss):
		return None

	# backward
	if not is_val:
		info.ops_backwarding(info, loss)

	# pack result and return
	return info.ops_pack_batch_result(info, x, y, predict, loss)


def trainEpoch(info: ModelInfo, dataset: Dataset, is_val=False) -> Any:
	# CONFIG
	result_list: List[Any] = []

	# get data loader
	dataloader: DataLoader = info.ops_get_dataloader(info, dataset)

	# CORE
	# get batch from loader and train it
	for data in tqdm.tqdm(dataloader):

		# to device - x
		if isinstance(data[0], list):
			x: List[torch.Tensor] = []
			for item in data[0]:
				x.append(item.to(info.device_current))
		else:
			x: torch.tensor = data[0].to(info.device_current)

		# to device - y
		if isinstance(data[1], list):
			y: List[torch.Tensor] = []
			for item in data[1]:
				y.append(item.to(info.device_current))
		else:
			y: torch.tensor = data[1].to(info.device_current)

		# actual train
		# ignore None return
		ret: Any = trainBatch(info, x, y, is_val)
		if ret is None:
			continue
		result_list.append(ret)

	# RET
	# pack result and return
	return info.ops_pack_epoch_result(info, result_list)


def _train_(info: ModelInfo, dataset: Dataset) -> None:
	# start
	info.model.train(True)
	info.device_current = info.device_train
	info.model 			= info.model.to(info.device_current)

	info.executeProcess(ModelInfo.Stage.TRAIN_START, {})
	info.stage = ModelInfo.Stage.TRAIN

	# actual
	ret: Any = trainEpoch(info, dataset, is_val=False)
	info.ops_handle_train_result(info, ret)

	# end
	info.stage = ModelInfo.Stage.TRAIN_END
	info.executeProcess(ModelInfo.Stage.TRAIN_END, {})


def _validate_(info: ModelInfo, dataset: Dataset) -> None:
	# start
	info.model.train(False)
	info.device_current 	= info.device_test
	info.model 				= info.model.to(info.device_current)

	info.executeProcess(ModelInfo.Stage.VAL_START, {})
	info.stage = ModelInfo.Stage.VAL

	# actual
	ret: Any = trainEpoch(info, dataset, is_val=True)
	info.ops_handle_validate_result(info, ret)

	# end
	info.stage = ModelInfo.Stage.VAL_END
	info.executeProcess(ModelInfo.Stage.VAL_END, {})


def _test_(info: ModelInfo, dataset: Dataset) -> None:
	# start
	info.model.train(False)
	info.device_current = info.device_test
	info.model 			= info.model.to(info.device_current)

	info.executeProcess(ModelInfo.Stage.TEST_START, {})
	info.stage = ModelInfo.Stage.TEST

	# actual
	ret: Any = trainEpoch(info, dataset, is_val=True)
	info.ops_handle_test_result(info, ret)

	# end
	info.stage = ModelInfo.Stage.TEST_END
	info.executeProcess(ModelInfo.Stage.TEST_END, {})


def train(info: ModelInfo) -> None:
	# start
	epoch: int = info.epoch

	# put model to correct place
	info.model = info.model.to(info.device_train)

	# signal
	info.executeProcess(ModelInfo.Stage.START, {})

	# train, validate, test
	#
	# train: sample of data used to fit the model
	#
	# validate: sample of data used to provide an unbiased evaluation of
	# 			an model fit on the training dataset while
	# 			tuning model hyper-parameters
	#
	# test: sample of data used to provide an unbiased evaluation of
	# 		a final model fit on the training dataset
	#
	# reference
	# - https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
	for i in range(epoch):

		# train, validate
		_train_(	info,	info.dataset["Train"])
		_validate_(	info,	info.dataset["Val"])

		# scheduler
		if info.scheduler is not None:
			info.scheduler.step()

		info.iteration += 1

	_test_(info, info.dataset["Test"])

	# end
	info.executeProcess(ModelInfo.Stage.END, {})
