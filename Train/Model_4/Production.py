import math
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
def Production_main() -> None:
	# load model info
	info = ModelInfo()

	with open(os.path.join(os.path.dirname(__file__), "ModelInfo.json"), "r") as f:
		data = json.load(f)
		info.setDictData(data)

	# load dataset
	path_data_train 	= info.dataset_info["Path_Data_Train"]
	path_data_validate 	= info.dataset_info["Path_Data_Val"]
	path_data_test 		= info.dataset_info["Path_Data_Test"]
	# path_ratio_train	= info.dataset_info["Path_Ratio_Train"]
	# path_ratio_validate	= info.dataset_info["Path_Ratio_Val"]
	# path_ratio_test		= info.dataset_info["Path_Ratio_Test"]
	path_image			= info.dataset_info["Path_Image"]

	dataset_train 		= Dataset_Train.loadDataset_CSV(	path_data_train,	path_image, "Train")
	dataset_validate 	= Dataset_Train.loadDataset_CSV(	path_data_validate,	path_image,	"Val")
	dataset_test 		= Dataset_Train.loadDataset_CSV(	path_data_test,		path_image,	"Test")

	# show image
	data 	= dataset_test[1]
	data_x	= data[0]
	image_x = data_x.reshape(*(data_x.shape[1:]))
	image_x = image_x.reshape((-1, image_x.shape[-2], image_x.shape[-1]))

	data_y	= data[1]
	image_y = data_y[1].reshape(*(data_y[1].shape[1:]))

	image = torch.cat([image_x, image_y], dim=0)

	size_image = image.shape[0]
	# size_col: int = math.floor(math.sqrt(size_image))
	size_col: int = 4
	size_row: int = size_image // size_col

	_, axis_list = plt.subplots(size_row, size_col)
	for i, axis in enumerate(axis_list.reshape(-1)):
		axis.imshow(image[i])

	plt.show()

	# run model
	model = Model_4()
	model.load_state_dict(torch.load("./Result/Stage_2/Result_20210321175108/StateDict.tar"))
	model = model.train(False)

	result = model(data[0], {})
	image = result[1].detach().cpu().numpy()  # image: shape: [B, C, D, H, W]
	image = image.reshape(*(image.shape[1:]))

	_, axis_list = plt.subplots(4, 4)
	for i, axis in enumerate(axis_list.reshape(-1)):
		axis.imshow(image[0][i])

	plt.show()

	# w, h, d = 53, 53, 16
	#
	# # order: [d, h, w]
	# # (d1, h1, w1), (d1, h1, w2), ..., (d1, h1, wn), (d1, h2, w1), ...
	# x = np.arange(0, w)
	# x = np.tile(x, h * d)
	#
	# y = np.arange(0, w)
	# y = np.repeat(y, w)
	# y = np.tile(y, d)
	#
	# z = np.arange(0, d)
	# z = np.repeat(z, h * w)
	#
	# d = image[0].reshape(-1)
	# d = (d + 1) / 2.0
	# # s = d * (d > 0.55)
	# s = d
	# s *= 100
	#
	# # plot
	# ax = plt.axes(projection="3d")
	# ax.scatter3D(x, y, z, s=s, c=d)
	# plt.show()


# Operation
# ...
