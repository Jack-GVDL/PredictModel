from typing import *
import time
from .DataList import SnapshotControl, DataBase, DataList
from .DataConversion import DataConversion
from .DataConversion import DataUpdateLog


class DataControl:

	def __init__(self):
		super().__init__()

		# data
		self._data_list: List[DataList] = []

		# format
		# List[Tuple[src_data, dst_data, conversion]]
		self._link_list: List[Tuple[List[DataList], DataList, DataConversion]] = []

		self.snapshot_control = SnapshotControl()

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	def addData(self, data: DataList) -> bool:
		# first check if data already exist in the list
		if data in self._data_list:
			return False

		# add to data_list
		self._data_list.append(data)

		# config DataList
		data.snapshot_control = self.snapshot_control

		return True

	def rmData(self, data: DataList) -> bool:
		# ----- check exist -----
		# get index of the data
		try:
			index: int = self._data_list.index(data)
		except ValueError:
			return False

		# ----- link list -----
		# remove from link list
		# get list of index that contain this data
		#
		# done in one line
		# https://stackoverflow.com/questions/28182569/get-all-indexes-for-a-python-list/28182579
		index_list: List[int] = [i for i, link in enumerate(self._link_list) if (link[0] is data or link[1] is data)]

		# need to reverse the list or otherwise will pop the wrong item
		index_list.reverse()

		for i in index_list:
			self._link_list.pop(i)

		# ----- data list -----
		# remove from data_list
		self._data_list.pop(index)

		return True

	def addLink(self, src: List[DataList], dst: DataList, conversion: DataConversion) -> bool:
		# for this function
		# assume: only one valid conversion path from src to dst

		# ----- add to data_list (if not exist) -----
		for data in src:
			self.addData(data)
		self.addData(dst)

		# ----- add to list -----
		# TODO: missing cyclic check
		self._link_list.append((src, dst, conversion))

		return True

	def rmLink(self, src_list: List[DataList], dst: DataBase) -> bool:
		# assume: only one item in link_list will match for both src and dst
		for index, link in enumerate(self._link_list):
			if not self._matchList_DataList_(src_list, link[0]) or link[1] != dst:
				continue

			# remove from list
			self._link_list.pop(index)
			return True

		return False

	def addComposite(self, src: List[DataList], dst: DataList, conversion_list: List[DataConversion]) -> bool:
		# ----- check -----
		if not conversion_list:
			return False

		# ----- add to data_list (if not exist in data_list) -----
		# src and dst
		for data in src:
			self.addData(data)
		self.addData(dst)

		# intermediate
		# intermediate_list: [src, [intermediate_1], [intermediate_2], ..., [dst]]
		intermediate_list: List[List[DataList]] = [src]

		for i in range(len(conversion_list) - 1):
			data_list = DataList()
			data_list.name		= "intermediate"
			data_list.data_key 	= conversion_list[i].key_dst
			self.addData(data_list)

			# data_list should be inside an array
			intermediate_list.append([data_list])

		# add dst to intermediate_list
		intermediate_list.append([dst])

		# ----- add link -----
		for i in range(len(conversion_list)):
			self._link_list.append((intermediate_list[i], intermediate_list[i + 1][0], conversion_list[i]))

		return True

	# operation
	def update(self, data: DataList, log_list: List[DataUpdateLog] = None) -> bool:
		# check if in the data_list
		if not (data in self._data_list):
			return False

		# actual update
		self._update_(data, log_list)

		return True

	# Protected
	# utility
	def _getLinkList_Src_(self, src: DataList) -> List[Tuple[List[DataList], DataList, DataConversion]]:
		# assumed: src must in data_list
		link_list: List[Tuple[List[DataList], DataList, DataConversion]] = []

		for link in self._link_list:
			if link[0] != src:
				continue
			link_list.append(link)

		return link_list

	def _getLinkList_Dst_(self, dst: DataList) -> List[Tuple[List[DataList], DataList, DataConversion]]:
		# assumed: dst must in data_list
		link_list: List[Tuple[List[DataList], DataList, DataConversion]] = []

		for link in self._link_list:
			if link[1] != dst:
				continue
			link_list.append(link)

		return link_list

	# operation
	def _update_(self, data: DataList, log_list: List[DataUpdateLog] = None) -> bool:
		# search method: dfs
		# assumed: no cycle

		# CONFIG
		is_updated: 		bool = False
		is_first_update:	bool = True

		# get link_list (that the data is dst)
		# it is reminded that data is the target and we want to search data will be converted to target
		link_list = self._getLinkList_Dst_(data)

		# load the data (if data not loaded)
		# if not data.load():
		# 	return False

		# CORE
		for link in link_list:

			# update recursively
			src_list: List[DataList] = link[0]
			for src in src_list:
				if not self._update_(src, log_list):
					continue

			# check if needed to update based on the time-snap between dst and src_list
			# then reset the data (dst) if needed
			# where dst will only be reset ONCE
			if not self._checkIsRequireUpdate_(src_list, data):
				continue
			if is_first_update:
				data.reset()
				is_first_update = False

			# prepare log object
			log: DataUpdateLog = DataUpdateLog()

			# start time
			log.start(time.time())

			# actual update
			# TODO: currently no exception handling
			result: bool = link[2].convert(link[0], data, log)

			# end time
			log.end(time.time())

			# log
			if log_list is not None:
				log_list.append(log)
			is_updated = True

		# if dst is updated, that mark it
		if is_updated:
			data.markUpdate()

		return True

	def _matchList_DataList_(self, list_1: List[DataList], list_2: List[DataList]) -> bool:
		# strict comparison, order must be the same
		if len(list_1) != len(list_2):
			return False

		# foreach DataList
		for index in range(len(list_1)):
			if list_1[index] == list_2[index]:
				continue
			return False

		return True

	def _checkIsRequireUpdate_(self, src_list: List[DataList], dst: DataList) -> bool:
		# check if data is static or volatile
		if dst.is_static:
			return False
		if dst.is_volatile:
			return True

		# TODO: this should be configurable
		if not src_list:
			return True

		# foreach src DataList
		for src in src_list:
			if not self._checkIsRequireUpdate_Single_(src, dst):
				continue
			return True
		return False

	def _checkIsRequireUpdate_Single_(self, src: DataList, dst: DataList) -> bool:
		# backup
		# check by time_snap
		# # update if src.time_snap > dst.time_snap
		# if src.time_snap[0] > dst.time_snap[0] or \
		# 	src.time_snap[1] > dst.time_snap[1] or \
		# 	src.time_snap[2] > dst.time_snap[2] or \
		# 	src.time_snap[3] > dst.time_snap[3] or \
		# 	src.time_snap[4] > dst.time_snap[4] or \
		# 	src.time_snap[5] > dst.time_snap[5]:
		# 	return True

		if src.snapshot <= dst.snapshot:
			return False

		return True
