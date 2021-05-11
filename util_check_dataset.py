"""
produce a list of valid data (in form of csv)

reason of invalid data
- for the given time and position, corresponding satellite image is absent

in Lawrence's code
validate set: 	subset_0
test set:		subset_1
train set:		subset_2, subset_3, subset_4
"""


from typing import *
from Source import *
import datetime
import os
from Train.Model_1.Dataset_Train import Dataset_Train


# Data
is_test:	bool 	= False
# path_image			= "../../Data/D_Image_S_Filename_E_Resolution_1/Data"
# path_image			= "../../Data/D_Image_S_Filename_E_Resolution_2/Data"
path_image			= "../../Data/D_Image_S_Filename_E_Resolution_2_Combined/Data"

source_image 		= "Filename_2"
source_data_list 	= "Height"
# source_data_list	= "Filename"

extra_info			= "_T_MINUS_40_T_0"

filename_src		= f"T_Turbulence_S_{source_data_list}_Subset"
filename_dst		= f"T_Turbulence_S_{source_image}_{source_data_list}_Subset"

offset_: List[int] = [-40, 0]


# Function
def getDatetime_Offset(date: List[int], time: List[int], offset: int) -> List[List[int]]:
	# offset: in minute

	time_current = datetime.datetime(
		year=date[0], month=date[1], day=date[2],
		hour=time[0], minute=time[1], second=0
	)
	time_target = time_current + datetime.timedelta(seconds=offset * 60)

	return [[time_target.year, time_target.month, time_target.day], [time_target.hour, time_target.minute]]


def getPath_Image(date: List[int], time: List[int], longitude: float, latitude: float) -> str:
	filename: str = \
		f"{date[0]:04d}{date[1]:02d}{date[2]:02d}_" \
		f"{time[0]:02d}{time[1]:02d}_" \
		f"{longitude}_{latitude}.npy"

	return os.path.join(path_image, filename)


def process(subset: int, offset: List[int]) -> None:
	path_src 		= f"../Database/{filename_src}_{subset}_D_CSV.csv"
	path_dst		= f"./Data/{filename_dst}_{subset}_D_CSV_E_Reduced{extra_info}.csv"

	# test: check if the newly created file is valid or not
	# if is valid, then all the data in the file should be valid
	#
	# also, this checking means that the creation of new file is not needed
	if is_test:
		path_src = path_dst

	# load dataset to get the data_list
	dataset 	= Dataset_Train.loadDataset_CSV(path_src, "", "Train", is_verbose=True)
	data_list 	= dataset._data_list
	data_key	= DataKey_Turbulence()

	# foreach data in data_list
	# check if the corresponding file is exist in dataset file
	size_exist:		int = 0
	size_absent: 	int = 0
	valid_list:		List[Any] = []

	for data in data_list:
		date 		= data[data_key.date]
		time 		= data[data_key.time]
		longitude 	= data[data_key.longitude]
		latitude 	= data[data_key.latitude]
		height		= data[data_key.height]
		turbulence	= data[data_key.turbulence]

		is_valid: bool = True
		for i in range(offset[0], offset[1] + 1, 10):

			datetime_ 	= getDatetime_Offset(date, time, i)
			path 		= getPath_Image(datetime_[0], datetime_[1], longitude, latitude)

			if os.path.isfile(path):
				continue

			is_valid = False
			break

		if not is_valid:
			size_absent += 1
			continue

		if not is_test:
			valid_list.append([date, time, longitude, latitude, height, turbulence])

		size_exist += 1

	# save to new csv
	if not is_test:

		with open(path_dst, "w") as f:
			for data in valid_list:
				f.write(
					f"{data[0][0]:04d}-{data[0][1]:02d}-{data[0][2]:02d},"
					f"{data[1][0]:02d}:{data[1][1]:02d},"
					f"{data[2]},{data[3]},{data[4]},{data[5]}\n")

	# message
	print(f"Exist: {size_exist}; Absent: {size_absent}")
	print()


def main() -> None:
	process(0, offset_)
	process(1, offset_)
	process(2, offset_)
	process(3, offset_)
	process(4, offset_)


# Operation
if __name__ != "__main__":
	raise NotImplementedError


main()
