from typing import *
import datetime
import torch
import random
import numpy as np
import json
import os
from Source import *
from .Linker_Train import File_Linker_Train
from .Linker_Train import Linker_Train
from .Dataset_Train import File_Dataset_Train
from .Dataset_Train import Dataset_Train


# Data Structure
class File_Main(Interface_CodePath):

	# Interface
	def getCodePath(self) -> str:
		return __file__


# Function
# util
def loadRatio(path: str) -> List[int]:
	with open(path, "r") as f:
		data = json.load(f)
	return data


# config
def configEnvironment(info: ModelInfo, is_loaded: bool = False) -> None:
	# device
	env_device = "cuda:0" if torch.cuda.is_available() else "cpu"
	env_device = torch.device(env_device)

	info.device_train 	= env_device
	info.device_test	= env_device

	# random seed
	#
	# reference
	# - https://pytorch.org/docs/stable/notes/randomness.html
	info.random_seed = 123
	torch.manual_seed(info.random_seed)
	random.seed(info.random_seed)
	np.random.seed(info.random_seed)

	# message
	print("----- Environment -----")
	print(Util_Torch.getString_Environment())
	print(f"Random seed: { info.random_seed }")
	print()


def configTrain(info: ModelInfo, is_loaded: bool = False) -> None:
	# train info, model
	info.model = Model_3()

	if not is_loaded:
		info.epoch 		= 0
		info.batch_size = 8

	# ops
	linker: Linker_Train = info.train_object["Linker"]

	info.ops_get_loss				= linker.Ops_getLoss
	info.ops_get_dataloader			= linker.Ops_getDataLoader
	info.ops_pack_batch_result		= linker.Ops_packBatchResult
	info.ops_pack_epoch_result		= linker.Ops_packEpochResult
	info.ops_handle_train_result	= linker.Ops_handleTrainResult
	info.ops_handle_validate_result	= linker.Ops_handleValidateResult
	info.ops_handle_test_result		= linker.Ops_handleTestResult

	# message
	print(f"----- Model -----")
	print(info.model)
	print(f"Epoch: { info.epoch }")
	print(f"Batch size: { info.batch_size }")
	print()


def configDataset(info: ModelInfo, is_loaded: bool = False) -> None:
	if not is_loaded:
		# set default path
		info.dataset_info["Path_Data_Train"] 	= "./Data/Data_Train.csv"
		info.dataset_info["Path_Data_Val"] 		= "./Data/Data_Validate.csv"
		info.dataset_info["Path_Data_Test"] 	= "./Data/Data_Test.csv"
		info.dataset_info["Path_Ratio_Train"]	= "./Data/Ratio_Train.json"
		info.dataset_info["Path_Ratio_Val"]		= "./Data/Ratio_Validate.json"
		info.dataset_info["Path_Ratio_Test"]	= "./Data/Ratio_Test.json"
		info.dataset_info["Path_Image"]			= "../../Data/D_Image_S_Filename/Data"

	# get path
	path_data_train 	= info.dataset_info["Path_Data_Train"]
	path_data_validate 	= info.dataset_info["Path_Data_Val"]
	path_data_test 		= info.dataset_info["Path_Data_Test"]
	path_ratio_train	= info.dataset_info["Path_Ratio_Train"]
	path_ratio_validate	= info.dataset_info["Path_Ratio_Val"]
	path_ratio_test		= info.dataset_info["Path_Ratio_Test"]
	path_image			= info.dataset_info["Path_Image"]

	# load dataset
	dataset_train 		= Dataset_Train.loadDataset_CSV(	path_data_train,	path_image, "Train")
	dataset_validate 	= Dataset_Train.loadDataset_CSV(	path_data_validate,	path_image,	"Val")
	dataset_test 		= Dataset_Train.loadDataset_CSV(	path_data_test,		path_image,	"Test")

	# dataset list
	info.dataset["Train"]	= dataset_train
	info.dataset["Val"]		= dataset_validate
	info.dataset["Test"]	= dataset_test

	# load ratio (for LDAM)
	info.dataset_object["Ratio_Train"]	= np.array(loadRatio(path_ratio_train))
	info.dataset_object["Ratio_Val"]	= np.array(loadRatio(path_ratio_validate))
	info.dataset_object["Ratio_Test"]	= np.array(loadRatio(path_ratio_test))

	# message
	print(f"----- Dataset -----")
	print(f"Path: data train:     {path_data_train}")
	print(f"Path: data validate:  {path_data_validate}")
	print(f"Path: data test:      {path_data_test}")
	print(f"Path: ratio train:    {path_ratio_train}")
	print(f"Path: ratio validate: {path_ratio_validate}")
	print(f"Path: ratio test:     {path_ratio_test}")
	print(f"Path: image:          {path_image}")
	print()


def configProcess(info: ModelInfo, is_loaded: bool = False) -> None:
	# CONFIG
	now 			= datetime.datetime.now()
	current_time 	= now.strftime("%Y%m%d%H%M%S")
	save_folder 	= f"Result_{current_time}"

	linker: Linker_Train = info.train_object["Linker"]

	# create
	process_fs_build	= TrainProcess_FileControlBuilder()
	process_fs_update	= TrainProcess_FileControlUpdater()
	process_file_python	= TrainProcess_PythonFile()
	process_file_json	= TrainProcess_DictSave()
	process_scheduler	= linker.process_scheduler
	process_result		= linker.process_result

	# print
	process_result.is_print = True
	process_result.is_log	= True

	# stage
	process_fs_build.addStage(		ModelInfo.Stage.START)
	process_file_python.addStage(	ModelInfo.Stage.START)
	process_file_json.addStage(		ModelInfo.Stage.START)
	process_fs_update.addStage(		ModelInfo.Stage.START)

	process_fs_update.addStage(		ModelInfo.Stage.VAL_END)
	process_fs_update.addStage(		ModelInfo.Stage.END)

	# config
	process_fs_build.setTargetPath("./Result", save_folder)
	process_file_python.addPythonFile(info.model, 			"Model")
	process_file_python.addPythonFile(File_Linker_Train(), 	"Linker_Train")
	process_file_python.addPythonFile(File_Dataset_Train(),	"Dataset_Train")
	process_file_python.addPythonFile(File_Main(),			"main")

	# process control
	info.process_control.addProcess(process_fs_build)
	info.process_control.addProcess(process_file_python)
	info.process_control.addProcess(process_file_json)
	info.process_control.addProcess(process_scheduler)
	info.process_control.addProcess(process_result)
	info.process_control.addProcess(process_fs_update)


def Model_main() -> None:
	# info, linker
	info 						= ModelInfo()
	linker 						= Linker_Train()
	info.train_object["Linker"] = linker

	# load from json
	is_load_from_json: 	bool	= True
	path_json: 			str 	= os.path.join(os.path.dirname(__file__), "./ModelInfo.json")

	if is_load_from_json:
		with open(path_json, "r") as f:
			data = json.load(f)
			info.setDictData(data)

	# config
	configEnvironment(	info,	is_load_from_json)
	configTrain(		info,	is_load_from_json)
	configDataset(		info,	is_load_from_json)
	configProcess(		info,	is_load_from_json)

	# train
	train(info)
