import math
import tqdm
from typing import *
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from Source import *
from .Dataset_Train import File_Dataset_Train
from .Dataset_Train import Dataset_Train


# Data Structure
# ...


# Function
def testDataset(info: ModelInfo, dataset: Dataset_Train, mask=None) -> List[Any]:
	result_predict: List[Any] = []
	result_y:		List[Any] = []

	# to cuda
	info.model = info.model.cuda()

	for data in tqdm.tqdm(dataset):
		x = data[0]
		y = data[1][0][0]

		# mask
		if mask is not None:
			x[1] = torch.tensor(mask)

		# to cuda
		x[0] = x[0].cuda()
		x[1] = x[1].cuda()

		# get forward result
		predict = info.model(x, {})

		predict			= predict[0]
		predict			= predict.cpu()

		predict_class 	= torch.argmax(predict[0][y[1]])

		result_predict.append(predict_class.int())
		result_y.append(y[0].int())

	return [result_predict, result_y]


def getResult(result: List[Any], size_class: int) -> None:
	matrix = computeConfusionMatrix(
		np.array(result[0]),
		np.array(result[1]),
		size_class
	)

	confusion_matrix = ConfusionMatrix(matrix)

	# result message
	# print("Accuracy")
	# for i in range(size_class):
	# 	print(f"class {i}: score: {confusion_matrix.getAccuracy_Single(i)}")
	#
	# print()

	print("F1 Score")
	for i in range(size_class):
		print(f"class {i}: score: {round(confusion_matrix.getF1Score_Single(i), 2)}")

	print()


def computeConfusionMatrix(predict: np.ndarray, y: np.ndarray, size_class: int) -> np.ndarray:
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


def Result_main() -> None:
	# load model info
	info = ModelInfo()

	with open(os.path.join(os.path.dirname(__file__), "ModelInfo.json"), "r") as f:
		data = json.load(f)
		info.setDictData(data)

	# load model
	model = Model_6()
	model.load_state_dict(torch.load("./Result/Stage_3/Result_20210329155629/StateDict.tar"))
	model = model.train(False)
	info.model = model

	# load dataset
	path_data_train 	= info.dataset_info["Path_Data_Train"]
	path_data_validate 	= info.dataset_info["Path_Data_Val"]
	path_data_test 		= info.dataset_info["Path_Data_Test"]
	path_image			= info.dataset_info["Path_Image"]

	dataset_train 		= Dataset_Train.loadDataset_CSV(	path_data_train,	path_image, "Train")
	dataset_validate 	= Dataset_Train.loadDataset_CSV(	path_data_validate,	path_image,	"Val")
	dataset_test 		= Dataset_Train.loadDataset_CSV(	path_data_test,		path_image,	"Test")

	# train
	result = testDataset(info, dataset_test, [True, False, True, True])

	# 3-class result
	print("3 - class")
	getResult(result, 3)

	# 2-class result (MOD, SEV to be the same class)
	print("2 - class")

	for i in range(len(result[0])):
		result[0][i] = min(result[0][i], 1)
		result[1][i] = min(result[1][i], 1)

	getResult(result, 2)
