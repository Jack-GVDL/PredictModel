from typing import *


class _DataKey_:

	def __init__(self, name: str):
		super().__init__()

		# data
		self._name: str = ""  # work as key, should be an unique name

		# operation
		self._name = name

	def __del__(self):
		return

	# Property
	# getter
	@property
	def name(self) -> str:
		return self._name

	# TODO: remove
	# @property
	# def index(self) -> int:
	# 	return self._index

	# setter
	@name.setter
	def name(self, name: str) -> None:
		self._name = name

	# TODO: remove
	# @index.setter
	# def index(self, index: int) -> None:
	# 	self._index = index

	# Operation
	# TODO: remove
	# def copy(self) -> Any:
	# 	key = DataKey()
	# 	key.name = self.name
	# 	return key

	# Operator Overloading
	# ...


class DataKey(_DataKey_):

	# singleton
	@classmethod
	def getDataKey(cls) -> _DataKey_:
		raise NotImplementedError

	def __init__(self, name: str = ""):
		super().__init__(name)
		
		# data
		self._key_list: List[_DataKey_] = []
		
		# operation
		# ...
		
	def __del__(self):
		return

	# Property
	@property
	def key_list(self) -> List[_DataKey_]:
		return self._key_list.copy()
		
	# Operation
	def reset(self) -> bool:
		self._key_list.clear()
		return True

	def addDataKey(self, key: _DataKey_) -> bool:
		self._key_list.append(key)
		return True

	def rmDataKey(self, key: _DataKey_) -> bool:
		try:
			index: int = self._key_list.index(key)
		except ValueError:
			return False

		self._key_list.pop(index)
		return True

	# by content in key instead of the object_id of key
	def getKeyIndex_Key(self, k: _DataKey_) -> int:
		for index, key in enumerate(self._key_list):
			if key != k:
				continue
			return index
		return -1

	def getKeyIndex_Name(self, name: str) -> int:
		for index, key in enumerate(self._key_list):
			if name != key.name:
				continue
			return index
		return -1

	# Protected
	# ...

	# Operator Overloading
	def __getitem__(self, index: int) -> _DataKey_:
		# if index < 0 or index >= len(self._key_list):
		# 	return None
		return self._key_list[index]

	def __setitem__(self, index: int, value: _DataKey_) -> None:
		if index < 0 or index >= len(self._key_list):
			return
		self._key_list[index] = value

	def __eq__(self, other: _DataKey_) -> bool:
		# DataKey are the same if either
		# - the same object
		# - the same content

		# ----- object -----
		# must not use self == other
		# == is an operator (operator overload)
		if self is other:
			return True

		# ----- content -----
		# if either one of the key is leaf node
		# then the method of comparing the child in key list is not applicable
		if not self._key_list or not other._key_list:
			return False

		# compare the key list one-by-one
		if len(self._key_list) != len(other._key_list):
			return False

		for index in range(len(self._key_list)):
			if self._key_list[index] == other._key_list[index]:
				continue
			return False

		return True

	def __str__(self) -> str:
		content: str = ""

		# self
		content += self._name
		content += '\n'

		# children
		for index, child in enumerate(self._key_list):
			content += str(child)
			# if index != len(self._key_list):
			# 	content += '\n'

		return content
