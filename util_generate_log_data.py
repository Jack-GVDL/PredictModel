"""
generate logData json file for training result
"""


from typing import *
from Source import *
import json
import numpy as np


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


def main() -> None:
	# CONFIG
	path_model_info = "./Result/Stage_4/Result_20210407132631/ModelInfo.json"
	path_data		= "./Result/Stage_4/Result_20210407132631/Data_ModelInfo.json"

	info 			= ModelInfo()
	control_data 	= Control_Data()

	# ----- model info -----
	# load
	with open(path_model_info, "r") as f:
		data = json.load(f)
		info.setDictData(data)

	# ----- compute training result -----
	iteration: 	int = info.result["Best_Iteration"]
	epoch:		int = min(
						len(info.result["Train_Loss"]),
						len(info.result["Val_Loss"]))

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

	# accuracy
	accuracy_train_list: 	List[List[float]] = getAccuracyList(matrix_train_list)
	accuracy_val_list:		List[List[float]] = getAccuracyList(matrix_val_list)

	# precision
	precision_train_list: 	List[List[float]] = getPrecisionList(matrix_train_list)
	precision_val_list:		List[List[float]] = getPrecisionList(matrix_val_list)

	# recall
	recall_train_list: 		List[List[float]] = getRecallList(matrix_train_list)
	recall_val_list:		List[List[float]] = getRecallList(matrix_val_list)

	# f1 score
	f1score_train_list: 	List[List[float]] = getF1ScoreList(matrix_train_list)
	f1score_val_list:		List[List[float]] = getF1ScoreList(matrix_val_list)

	# ----- log data -----
	control_data.addLog_Data(loss_train, 	Log_Data.DataType.FLOAT, "Loss/Train")
	control_data.addLog_Data(loss_val, 		Log_Data.DataType.FLOAT, "Loss/Val")

	for i in range(len(accuracy_train_list)):
		control_data.addLog_Data(accuracy_train_list[i], Log_Data.DataType.FLOAT, "Accuracy/Train/" + str(i))

	for i in range(len(accuracy_train_list)):
		control_data.addLog_Data(accuracy_val_list[i], Log_Data.DataType.FLOAT, "Accuracy/Val/" + str(i))

	for i in range(len(accuracy_train_list)):
		control_data.addLog_Data(precision_train_list[i], Log_Data.DataType.FLOAT, "Precision/Train/" + str(i))

	for i in range(len(accuracy_train_list)):
		control_data.addLog_Data(precision_val_list[i], Log_Data.DataType.FLOAT, "Precision/Val/" + str(i))

	for i in range(len(accuracy_train_list)):
		control_data.addLog_Data(recall_train_list[i], Log_Data.DataType.FLOAT, "Recall/Train/" + str(i))

	for i in range(len(accuracy_train_list)):
		control_data.addLog_Data(recall_val_list[i], Log_Data.DataType.FLOAT, "Recall/Val/" + str(i))

	for i in range(len(accuracy_train_list)):
		control_data.addLog_Data(f1score_train_list[i], Log_Data.DataType.FLOAT, "F1Score/Train/" + str(i))

	for i in range(len(accuracy_train_list)):
		control_data.addLog_Data(f1score_val_list[i], Log_Data.DataType.FLOAT, "F1Score/Val/" + str(i))

	# dump
	data = control_data.getDictData()
	data = json.dumps(data, indent=None, separators=(',', ':'))

	with open(path_data, "w") as f:
		f.write(data)


# Operation
if __name__ != "__main__":
	raise NotImplementedError


main()
