from typing import *
import datetime
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from Source import *


# Data Structure
class File_Dataset_Train(Interface_CodePath):

	# Interface
	def getCodePath(self) -> str:
		return __file__


# Data
_transform_train = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.RandomChoice([
		# transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		# transforms.RandomResizedCrop(224, scale=(0.85, 1.0))
	]),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.485, 0.456, 0.406], [0.229, 0.229, 0.224, 0.225])
])

_transform_validate = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.485, 0.456, 0.406], [0.229, 0.229, 0.224, 0.225])
])

_transform_test = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.485, 0.456, 0.406], [0.229, 0.229, 0.224, 0.225])
])

_transform = {
	"Train": _transform_train,
	"Val": _transform_validate,
	"Test": _transform_test
}


# Data Structure
class Dataset_Train(Dataset_Base):

	@classmethod
	# load dataset from csv
	def loadDataset_CSV(cls, path_csv: str, path_image: str, type_: str, is_verbose: bool = True) -> Dataset_Base:
		# CHECK
		if path_csv is None:
			return None

		# CORE
		# data
		data_list = DataList(data_key=DataKey_Turbulence(), name="Target")

		# handler
		handler = DataHandler_Text()
		handler.data_key = DataKey_Turbulence.getDataKey()
		handler.separator = "\n"
		handler.file_text.file_list = [path_csv]
		handler.func_load = cls._load_data_csv_

		# conversion
		conversion_loader = DataConversion_Loader()
		conversion_loader.setHandler(handler)

		# control
		control = DataControl()
		control.addComposite(
			[],
			data_list,
			[conversion_loader]
		)

		# load
		log_list: List[DataUpdateLog] = []

		if not control.update(data_list, log_list=log_list):
			return None

		# verbose
		if is_verbose:
			# printer
			control_content = Control_StringContent()
			handler_log = Handler_DataUpdateLog()

			handler_log.log_list = log_list
			handler_log.control_content = control_content
			handler_log.convert()

			# print
			printer_content = Printer_StringContent()
			printer_content.control_content = control_content

			printer_content.print()
			print("\n\n", end="")

		# RET
		return Dataset_Train(data_list, path_image, _transform[type_])

	# load data from csv
	@classmethod
	def _load_data_csv_(cls, data: List[Any], content: str) -> bool:
		# list of separated item
		item_list: List[str] = content.split(",")

		# structure of DataKey_Turbulence
		# date
		# time
		# longitude
		# latitude
		# height
		# turbulence
		data.append([int(item_list[0][:4]), int(item_list[0][5:7]), int(item_list[0][8:])])
		data.append([int(item_list[1][:2]), int(item_list[1][3:])])
		data.append(float(item_list[2]))
		data.append(float(item_list[3]))
		data.append(int(item_list[4]))
		data.append(int(item_list[5]))

		return True

	def __init__(self, data_list: DataList, path_image: str, transform: Any):
		super().__init__(data_list)

		# data
		self._path_image: str = path_image
		self._key_turbulence = DataKey_Turbulence()

		self._transform = transform

	# operation
	# ...

	def __del__(self):
		return

	# Property
	@property
	def path_image(self) -> str:
		return self._path_image

	# Operation
	# ...

	# Protected
	def _computeSingleData_(self, data: DataBase) -> Any:
		# CONFIG
		date: 			List[int]	= data[self._key_turbulence.date]
		time: 			List[int]	= data[self._key_turbulence.time]
		longitude: 		float 		= data[self._key_turbulence.longitude]
		latitude: 		float 		= data[self._key_turbulence.latitude]
		turbulence: 	int 		= data[self._key_turbulence.turbulence]
		height: 		int 		= data[self._key_turbulence.height]

		# CORE
		# time slice
		# -30 min to -20 min, 10 min interval
		datetime_list: List[List[List[int]]] = [
			self._getDatetime_Offset_(date, time, -20),
			self._getDatetime_Offset_(date, time, -10)
		]

		# read image
		image_list: List[torch.Tensor] = []
		for item in datetime_list:
			# to numpy (from npy)
			path_image: str = self._getPath_Image_(item[0], item[1], longitude, latitude)
			image: torch.Tensor = self._getImage_(path_image)
			image = image.reshape((1, 1, *image.shape))  # [B, T, C, H, W]

			# to tensor
			image_list.append(image)

		result_x: torch.Tensor = torch.cat(image_list, dim=1)

		# get severity
		# level of height: 16
		# range of height: 0 - 40,000, i.e. 2500ft per level
		level: int 	= height // 2500
		level 		= min(level, 15)

		result_y: torch.Tensor 	= torch.tensor([turbulence, level])
		result_y				= result_y.reshape((-1, 2))

		return result_x, result_y

	def _getPath_Image_(self, date: List[int], time: List[int], longitude: float, latitude: float) -> str:
		filename: str = \
			f"{date[0]:04d}{date[1]:02d}{date[2]:02d}_" \
			f"{time[0]:02d}{time[1]:02d}_" \
			f"{longitude}_{latitude}.npy"

		return os.path.join(self._path_image, filename)

	def _getImage_(self, path: str) -> torch.Tensor:
		# output format: [c, h, w], normalized
		# try to load image from disk
		try:
			image = np.load(path)
		except EOFError:
			return None

		# normalize
		for i in range(image.shape[0]):
			value_range = np.ptp(image[i, :, :])

			if value_range == 0:
				image[i, :, :] = np.zeros((image.shape[1], image.shape[2]))
			else:
				image[i, :, :] = (image[i, :, :] - np.min(image[i, :, :])) / value_range

		# transpose
		# from [C, H, W] to [H, W, C]
		image = image.transpose((1, 2, 0))

		# transform
		image = np.uint8(255 * image)
		image = Image.fromarray(image)
		image = self._transform(image)

		return image

	def _getDatetime_Offset_(self, date: List[int], time: List[int], offset: int) -> List[List[int]]:
		# offset: in minute

		time_current = datetime.datetime(
			year=date[0], month=date[1], day=date[2],
			hour=time[0], minute=time[1], second=0
		)
		time_target = time_current + datetime.timedelta(seconds=offset * 60)

		return [[time_target.year, time_target.month, time_target.day], [time_target.hour, time_target.minute]]
