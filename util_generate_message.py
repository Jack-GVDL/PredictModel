"""
generate message (text) based on the result
"""


from typing import *
from Source import *
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


# Data
path_result = "./Result/Stage_4/Result_20210407175343"


# Function
def main() -> None:
	# load model info
	info = ModelInfo()

	with open(os.path.join(path_result, "ModelInfo.json"), "r") as f:
		data = json.load(f)
		info.setDictData(data)

	# confusion matrix
	matrix = info.result["Test_Matrix"]
	confusion_matrix = ConfusionMatrix(np.array(matrix[0]))

	# message
	for i in range(3):
		print(f"class {i}: F1 score: {round(confusion_matrix.getF1Score_Single(i), 5)}")

	# 2-class confusion matrix
	matrix = np.array(matrix[0])
	matrix[1, :] += matrix[2, :]
	matrix[:, 1] += matrix[:, 2]
	matrix = matrix[:-1, :-1]
	confusion_matrix = ConfusionMatrix(matrix)

	# message
	for i in range(2):
		print(f"class {i}: F1 score: {round(confusion_matrix.getF1Score_Single(i), 5)}")


# Operation
if __name__ != '__main__':
	raise NotImplementedError

main()
