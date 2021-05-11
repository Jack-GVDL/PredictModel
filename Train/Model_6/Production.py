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
	path_image			= info.dataset_info["Path_Image"]

	dataset_train 		= Dataset_Train.loadDataset_CSV(	path_data_train,	path_image, "Train")
	dataset_validate 	= Dataset_Train.loadDataset_CSV(	path_data_validate,	path_image,	"Val")
	dataset_test 		= Dataset_Train.loadDataset_CSV(	path_data_test,		path_image,	"Test")

	# show image
	data 	= dataset_test[2]

	data_x	= data[0]
	image_x = data_x[0]
	image_x = image_x.reshape(*(image_x.shape[1:]))
	image_x = image_x.reshape((-1, image_x.shape[-2], image_x.shape[-1]))

	data_y	= data[1]
	image_y = data_y[1][-1]
	image_y = image_y.reshape((-1, image_y.shape[-2], image_y.shape[-1]))

	# image = torch.cat([image_x, image_y], dim=0)
	image = image_x

	size_image = image.shape[0]
	# size_col: int = math.floor(math.sqrt(size_image))
	size_col: int = 4
	size_row: int = size_image // size_col

	_, axis_list = plt.subplots(size_row, size_col)
	for i, axis in enumerate(axis_list.reshape(-1)):
		axis.imshow(image[i])

	plt.show()

	# show ground truth
	print(data_y[0])

	# config mask
	data_x[1][0] = True
	data_x[1][1] = True
	data_x[1][2] = True
	data_x[1][3] = True
	# data_x[1] = torch.cat([data[0][1], torch.tensor([False for _ in range(20)])])
	# print(data[0][1])

	# run model
	model = Model_6()
	model.load_state_dict(torch.load("./Result/Stage_3/Result_20210330171746/StateDict.tar"))
	model = model.train(False)

	result = model(data[0], {})
	image = result[1].detach().cpu().numpy()  # image: shape: [B, T, C, D, H, W]

	# time_slice = image.shape[1]
	# for t in range(time_slice):
	# 	print(t)
	#
	# 	image_cur = image[0, t, :, :, :, :]
	#
	# 	_, axis_list = plt.subplots(4, 4)
	# 	for i, axis in enumerate(axis_list.reshape(-1)):
	# 		axis.imshow(image_cur[0][i])
	#
	# 	plt.show()

	image = image[0, -1, :, :, :, :]
	image = image.reshape((-1, *(image.shape[2:])))

	_, axis_list = plt.subplots(4, 16)
	for i, axis in enumerate(axis_list.reshape(-1)):
		axis.imshow(image[i])
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
	# s = d * (d > 0.55)
	# # s = d
	# s *= 100
	#
	# # plot
	# ax = plt.axes(projection="3d")
	# ax.scatter3D(x, y, z, s=s, c=d)
	# plt.show()


# Operation
# ...
