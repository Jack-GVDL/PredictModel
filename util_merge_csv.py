"""
produce train, validate, test dataset (in form of csv)
"""


from typing import *
from Source import *
import os


# Data
train_subset_list:		List[int] = [2, 3, 4]
validate_subset_list:	List[int] = [1]
test_subset_list:		List[int] = [0]

path_src: str = f"./Data"

source_image 		= "Filename_2"
source_data_list 	= "Height"
# source_data_list 	= "Filename"
extra_info			= "_T_MINUS_40_T_0"

filename_src		= f"T_Turbulence_S_{source_image}_{source_data_list}_Subset"


# Function
def mergeDataset(subset_list: List[int], filename: str) -> None:
	# get path
	path_subset_list: List[str] = []
	for subset in subset_list:
		path = f"{filename_src}_{subset}_D_CSV_E_Reduced{extra_info}.csv"
		path = os.path.join(path_src, path)
		path_subset_list.append(path)

	# merge data
	with open(os.path.join(path_src, filename), "w") as f_dst:

		# foreach src file
		for path_subset in path_subset_list:
			with open(path_subset, "r") as f_src:

				for line in f_src:
					f_dst.write(line)

			# message
			print(f"Merge: {path_subset} -> {filename}")


def main() -> None:
	if not extra_info:
		dst_train	= f"Data_Train_S_{source_image}_{source_data_list}.csv"
		dst_val		= f"Data_Valid_S_{source_image}_{source_data_list}.csv"
		dst_test	= f"Data_Test_S_{source_image}_{source_data_list}.csv"

	else:
		dst_train	= f"Data_Train_S_{source_image}_{source_data_list}_E{extra_info}.csv"
		dst_val		= f"Data_Valid_S_{source_image}_{source_data_list}_E{extra_info}.csv"
		dst_test	= f"Data_Test_S_{source_image}_{source_data_list}_E{extra_info}.csv"

	mergeDataset(train_subset_list, 	dst_train)
	mergeDataset(validate_subset_list, 	dst_val)
	mergeDataset(test_subset_list, 		dst_test)


# Operation
if __name__ != "__main__":
	raise NotImplementedError


main()
