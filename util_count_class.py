"""
count the number of data in each class
"""


from typing import *
from Source import *
import os
import json
from Train.Model_1.Dataset_Train import Dataset_Train


# Data
size_class: int 	= 3
path_data: 	str 	= f"./Data"

source_image 		= "Filename_2"
# source_data_list 	= "Filename"
source_data_list 	= "Height"

extra_info			= "_E_T_MINUS_40_T_0"
# extra_info			= ""


# Function
def countClass(filename_src: str, filename_save: str = "", is_save: bool = False) -> None:
	# CONFIG
	table_class: List[int] = [0 for _ in range(size_class)]

	# CORE
	# get dataset
	dataset = Dataset_Train.loadDataset_CSV(os.path.join(path_data, filename_src), "", "Train", is_verbose=False)
	data_list = dataset._data_list
	data_key = DataKey_Turbulence()

	# count class
	for data in data_list:
		table_class[data[data_key.turbulence]] += 1

	# save to file
	if is_save:
		with open(os.path.join(path_data, filename_save), "w") as f:
			json.dump(table_class, f)

	# message
	print(f"File: {filename_src}")
	for i in range(size_class):
		print(f"Class {i}: {table_class[i]}")
	print()


def main() -> None:
	countClass(f"Data_Train_S_{source_image}_{source_data_list}{extra_info}.csv", 	f"Ratio_Train_S_{source_image}_{source_data_list}{extra_info}.json",	True)
	countClass(f"Data_Valid_S_{source_image}_{source_data_list}{extra_info}.csv", 	f"Ratio_Valid_S_{source_image}_{source_data_list}{extra_info}.json",	True)
	countClass(f"Data_Test_S_{source_image}_{source_data_list}{extra_info}.csv", 	f"Ratio_Test_S_{source_image}_{source_data_list}{extra_info}.json",		True)


# Operation
if __name__ != "__main__":
	raise RuntimeError


main()
