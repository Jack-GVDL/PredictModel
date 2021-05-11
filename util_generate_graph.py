"""
generate graph based on the result
"""


from typing import *
from Source import *
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Data
path_result = "./Result/Stage_4/Result_20210407175343"
# path_result = "./Result/Result_20210325230510"


# Function
def getAccuracyList(matrix_list: List[ConfusionMatrix]) -> List[List[float]]:
	# CHECK
	if not matrix_list:
		return []

	# CONFIG
	size_class: int 				= matrix_list[0].getSizeClass()
	result: 	List[List[float]] 	= [[] for _ in range(size_class + 1)]

	# CORE
	for matrix in matrix_list:
		result[0].append(matrix.getAccuracy())

		for i in range(size_class):
			result[i + 1].append(matrix.getAccuracy_Single(i))

	# RET
	return result


def getPrecisionList(matrix_list: List[ConfusionMatrix]) -> List[List[float]]:
	# CHECK
	if not matrix_list:
		return []

	# CONFIG
	size_class: int 				= matrix_list[0].getSizeClass()
	result: 	List[List[float]] 	= [[] for _ in range(size_class + 1)]

	# CORE
	for matrix in matrix_list:
		result[0].append(matrix.getAccuracy())

		for i in range(size_class):
			result[i + 1].append(matrix.getPrecision_Single(i))

	# RET
	return result


def getRecallList(matrix_list: List[ConfusionMatrix]) -> List[List[float]]:
	# CHECK
	if not matrix_list:
		return []

	# CONFIG
	size_class: int 				= matrix_list[0].getSizeClass()
	result: 	List[List[float]] 	= [[] for _ in range(size_class + 1)]

	# CORE
	for matrix in matrix_list:
		result[0].append(matrix.getAccuracy())

		for i in range(size_class):
			result[i + 1].append(matrix.getRecall_Single(i))

	# RET
	return result


def getF1ScoreList(matrix_list: List[ConfusionMatrix]) -> List[List[float]]:
	# CHECK
	if not matrix_list:
		return []

	# CONFIG
	size_class: int 				= matrix_list[0].getSizeClass()
	result: 	List[List[float]] 	= [[] for _ in range(size_class + 1)]

	# CORE
	for matrix in matrix_list:
		result[0].append(matrix.getAccuracy())

		for i in range(size_class):
			result[i + 1].append(matrix.getF1Score_Single(i))

	# RET
	return result


def plotConfusionMatrix(matrix: np.ndarray, path: str) -> None:
	# normalize if needed
	# sample_sum 	= np.sum(matrix, axis=1)
	# matrix 		= matrix.astype(np.float32)
	# matrix 		/= sample_sum

	# plot graph
	data_frame = pd.DataFrame(matrix, index=["0", "1", "2"], columns=["0", "1", "2"])
	plt.figure(figsize=matrix.shape)
	sn.heatmap(data_frame, annot=True, fmt="")

	# show and save graph
	# plt.show()
	plt.savefig(path, bbox_inches="tight")
	plt.clf()


def plotGraph_Line(
	data_list: List[Any], label_list: List[str], path: str,
	x_label: str, y_label: str, y_lim: float = None
	) -> None:

	# plot line
	plt.figure(figsize=(8, 8))

	if y_lim is not None:
		plt.ylim((0, y_lim))

	line_list: List[Any] = []

	for accuracy in data_list:
		line, = plt.plot(accuracy)
		line_list.append(line)

	# plot graph
	plt.ylabel(x_label)
	plt.xlabel(y_label)
	plt.legend(line_list, label_list, loc="upper right")

	# show and save graph
	# plt.show()
	plt.savefig(path, bbox_inches="tight")
	plt.clf()


def plotLoss(data_list: List[Any], label_list: List[str], path: str, y_lim=15.0) -> None:
	plotGraph_Line(data_list, label_list, path, "Loss", "Iteration", y_lim=y_lim)


def plotAccuracy(data_list: List[Any], label_list: List[str], path: str) -> None:
	plotGraph_Line(data_list, label_list, path, "Accuracy", "Iteration", y_lim=1.0)


def plotPrecision(data_list: List[Any], label_list: List[str], path: str) -> None:
	plotGraph_Line(data_list, label_list, path, "Precision", "Iteration", y_lim=1.0)


def plotRecall(data_list: List[Any], label_list: List[str], path: str) -> None:
	plotGraph_Line(data_list, label_list, path, "Recall", "Iteration", y_lim=1.0)


def plotF1Score(data_list: List[Any], label_list: List[str], path: str) -> None:
	plotGraph_Line(data_list, label_list, path, "F1 Score", "Iteration", y_lim=1.0)


def main() -> None:
	# load model info
	info = ModelInfo()

	with open(os.path.join(path_result, "ModelInfo.json"), "r") as f:
		data = json.load(f)
		info.setDictData(data)

	# ----- get the best result -----
	iteration: 	int = info.result["Best_Iteration"]
	epoch:		int = min(
						len(info.result["Train_Loss"]),
						len(info.result["Val_Loss"]))

	# confusion matrix
	matrix_val = info.result["Val_Matrix"][iteration]
	matrix_val = np.array(matrix_val)

	matrix_test = info.result["Test_Matrix"][0]
	matrix_test = np.array(matrix_test)

	# confusion matrix list
	matrix_val_list 	= []
	matrix_train_list 	= []

	for i in range(epoch):
		temp = info.result["Val_Matrix"][i]
		temp = ConfusionMatrix(np.array(temp))
		matrix_val_list.append(temp)

		temp = info.result["Train_Matrix"][i]
		temp = ConfusionMatrix(np.array(temp))
		matrix_train_list.append(temp)

	# loss
	loss_train 	= info.result["Train_Loss"]
	loss_val 	= info.result["Val_Loss"]

	# ----- plot -----
	# confusion matrix
	plotConfusionMatrix(
		matrix_val,
		os.path.join(path_result, "Matrix_Val.png"))

	plotConfusionMatrix(
		matrix_test,
		os.path.join(path_result, "Matrix_Test.png"))

	# loss
	plotLoss(
		[loss_train, loss_val],
		["Train", "Val"],
		os.path.join(path_result, "Loss.png"))

	plotLoss(
		[loss_train],
		["Train"],
		os.path.join(path_result, "Loss_Train.png"))

	plotLoss(
		[loss_val],
		["Val"],
		os.path.join(path_result, "Loss_Val.png"),
		y_lim=1.0
	)

	# accuracy
	accuracy_train_list: 	List[List[float]] = getAccuracyList(matrix_train_list)
	accuracy_val_list:		List[List[float]] = getAccuracyList(matrix_val_list)

	plotAccuracy(
		[accuracy_train_list[0], accuracy_val_list[0]],
		["Train", "Val"],
		os.path.join(path_result, "Accuracy_TrainVal.png"))

	plotAccuracy(
		accuracy_train_list[1:],
		["0", "1", "2"],
		os.path.join(path_result, "Accuracy_Train.png"))

	plotAccuracy(
		accuracy_val_list[1:],
		["0", "1", "2"],
		os.path.join(path_result, "Accuracy_Val.png"))

	# precision
	precision_train_list: 	List[List[float]] = getPrecisionList(matrix_train_list)
	precision_val_list:		List[List[float]] = getPrecisionList(matrix_val_list)

	plotPrecision(
		precision_train_list[1:],
		["0", "1", "2"],
		os.path.join(path_result, "Precision_Train.png"))

	plotPrecision(
		precision_val_list[1:],
		["0", "1", "2"],
		os.path.join(path_result, "Precision_Val.png"))

	# recall
	recall_train_list: 		List[List[float]] = getRecallList(matrix_train_list)
	recall_val_list:		List[List[float]] = getRecallList(matrix_val_list)

	plotRecall(
		recall_train_list[1:],
		["0", "1", "2"],
		os.path.join(path_result, "Recall_Train.png"))

	plotRecall(
		recall_val_list[1:],
		["0", "1", "2"],
		os.path.join(path_result, "Recall_Val.png"))

	# f1 score
	f1score_train_list: 	List[List[float]] = getF1ScoreList(matrix_train_list)
	f1score_val_list:		List[List[float]] = getF1ScoreList(matrix_val_list)

	plotF1Score(
		f1score_train_list[1:],
		["0", "1", "2"],
		os.path.join(path_result, "F1Score_Train.png"))

	plotF1Score(
		f1score_val_list[1:],
		["0", "1", "2"],
		os.path.join(path_result, "F1Score_Val.png"))


# Operation
if __name__ != "__main__":
	raise RuntimeError


main()
