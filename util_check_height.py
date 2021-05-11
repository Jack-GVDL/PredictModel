from typing import *
from Source import *
import datetime
import os
from Train.Model_6.Dataset_Train import Dataset_Train


# Data
# ...


# Function
def process(size_height: int, path: str) -> None:
	# CONFIG
	table_height: List[int] = [0 for _ in range(size_height)]

	# CORE
	# get dataset
	dataset = Dataset_Train.loadDataset_CSV(path, "", "Train", is_verbose=False)
	data_list = dataset._data_list
	data_key = DataKey_Turbulence()

	# count height
	for data in data_list:
		height = data[data_key.height]
		height = height // 5000
		height = min(height, size_height - 1)

		table_height[height] += 1

	# print message
	for i in range(size_height):
		print(f"Level: {i}: size sample: {table_height[i]}")


def main() -> None:
	print("Train")
	process(8, f"./Data/Data_Train_S_Filename_2_Height.csv")
	print()

	print("Val")
	process(8, f"./Data/Data_Valid_S_Filename_2_Height.csv")
	print()

	print("Test")
	process(8, f"./Data/Data_Test_S_Filename_2_Height.csv")
	print()


# Operation
if __name__ != "__main__":
	raise RuntimeError


main()
