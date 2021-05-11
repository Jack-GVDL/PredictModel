from typing import *
from ..DataChain import *
from .Data_PIREP import DataKey_PIREP
from .Data_Turbulence import DataKey_Turbulence


class DataConversion_PIREP_Turbulence(DataConversion):

	def __init__(self):
		super().__init__()

		# data
		self._name			= "PIREP_Turbulence"
		self._key_src_list 	= [DataKey_PIREP()]
		self._key_dst		= DataKey_Turbulence()

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	# ...

	# Protected
	def _setType_(self, key_src_list: List[DataKey], key_dst: DataKey) -> bool:
		raise RuntimeError("src_key_list and key_dst cannot be configured")

	def _convert_(self, src_list: List[DataList], dst: DataList, log: DataUpdateLog) -> bool:
		# CHECK
		if not src_list:
			log.log(src_list, dst, self, DataUpdateLog.State.FAIL)
			return False

		# CONFIG
		count_input: 	int = 0
		count_output:	int = 0

		# one-to-one conversion / mapping
		for src in src_list:
			for data in src:
				self._convertSingle_(data, dst)
			count_input += len(src)

		# stat
		count_output += len(dst)

		# log
		log.log(src_list, dst, self, DataUpdateLog.State.SUCCESS, count_input, count_output)

		return True

	def _convertSingle_(self, src: DataBase, dst: DataList) -> None:
		# CONFIG
		src_key: 	DataKey_PIREP 		= self._key_src_list[0]
		dst_key:	DataKey_Turbulence	= self._key_dst

		# allocate new data space
		data: DataBase = dst.createData()

		# direct mapping
		data[dst_key.date] 		= src[src_key.date]
		data[dst_key.time]		= src[src_key.time]
		data[dst_key.latitude]	= src[src_key.latitude]
		data[dst_key.longitude]	= src[src_key.longitude]
		data[dst_key.height]	= src[src_key.height]

		# conversion - turbulence
		# turbulence intensity
		# 0 = nil
		# 1 = light / Unknown
		# 2 = light to Moderate
		# 3 = moderate
		# 4 = moderate to Severe
		# 5 = severe
		#
		# TODO: need to check
		# turbulence
		# nil = 0, 1
		# mod = 2, 3
		# sev = 4, 5
		data[dst_key.turbulence] = src[src_key.pirep_turbulence] % 2
