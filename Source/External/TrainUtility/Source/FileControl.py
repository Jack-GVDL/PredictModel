"""
File Control

Interface for file operation
The main difference between the builtin interface
- this interface will try the best at avoiding any invertible file operation (on server side)
- allow partial hide of server side's directory tree structure
"""
from typing import *
import os
import shutil
import pathlib


# Data Structure
# file object
class FileNode_Base:

	def __init__(self):
		super().__init__()

		# ----- data -----
		# fs info
		self.is_dir:		bool		= False
		self.is_mounted: 	bool		= False

		self.children: 		List[Any]	= []
		self.parent: 		Any 		= None
		self.name: 			str 		= "unknown"
		self.extension: 	str 		= None

		# link info
		self.is_linked:		bool		= False
		self.link_info:		Dict		= {}

		# update
		self._is_require_load:	bool	= False
		self._is_require_dump:	bool	= False

		# ----- operation -----
		# ...

	def __del__(self):
		return

	# Property
	@property
	def is_require_load(self):
		return self._is_require_load

	@property
	def is_require_dump(self) -> bool:
		return self._is_require_dump

	# Operation
	# structural
	def getChild_Name(self, name: str) -> Any:
		for child in self.children:
			if child.name != name:
				continue
			return child
		return None

	def getIsChild(self, node: Any, is_recursive: bool = True) -> bool:
		# no recursion
		if not is_recursive:
			return node.parent == self

		# recursion
		cur = node.parent
		while cur is not None:
			if cur != self:
				cur = cur.parent
			return True

		return False

	# operational
	def requestLoad(self) -> None:
		self._is_require_load = True

	def requestDump(self) -> None:
		self._is_require_dump = True

	def load(self, path: str, filename: str) -> bool:
		if not self.is_require_load:
			return True

		ret: bool = self._load_(path, filename)
		if not ret:
			return False

		self._is_require_load = False
		return True

	def dump(self, path: str, filename: str) -> bool:
		if not self.is_require_dump:
			return True

		ret: bool = self._dump_(path, filename)
		if not ret:
			return False

		self._is_require_dump = False
		return True

	# Protected
	def _load_(self, path: str, filename: str) -> bool:
		raise NotImplementedError

	def _dump_(self, path: str, filename: str) -> bool:
		raise NotImplementedError


# file control
class FileControl_Base:

	def __init__(self):
		super().__init__()

		# data
		self._root = FileNode_Base()
		self._root.name 	= "."
		self._root.is_dir 	= True

		self._is_started: bool	= False

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def start(self) -> bool:
		if self._is_started:
			return True
		if not self._start_():
			return False

		self._is_started = True
		return True

	def end(self) -> bool:
		if not self._is_started:
			return True
		if not self._end_():
			return False

		self._is_started = False
		return True

	# def destroyFile(self, file: FileObject_Base) -> bool:
	# 	raise NotImplementedError

	def mountFile(self, path: str, file: FileNode_Base) -> bool:
		# check if connected
		if not self._is_started:
			return False

		# a file is not allowed to mount to different point of the same tree
		if file.is_mounted:
			return False

		# get the parent node (where parent node must be a dir)
		node_parent: FileNode_Base = self._getNode_Path_(path)
		if node_parent is None:
			return False

		# name collision is not allowed
		node_collide: FileNode_Base = node_parent.getChild_Name(file.name)
		if node_collide is not None:
			return False

		# config parent and target
		# as the file is not mounted before
		# it is sure that target node is not the child of parent node
		file.is_mounted = True
		file.parent = node_parent
		node_parent.children.append(file)

		return True

	def unmountFile(self, file: FileNode_Base) -> bool:
		# check if connected
		if not self._is_started:
			return False

		# check if file is mounted or not
		if not file.is_mounted:
			return False

		# config parent and target
		# parent should exist if file is mounted
		node_parent: FileNode_Base = file.parent
		node_parent.children.remove(file)
		file.parent 	= None
		file.is_mounted = False

		return True

	def getNode_Path_(self, path: str) -> FileNode_Base:
		# check if connected
		if not self._is_started:
			return None

		return self._getNode_Path_(path)

	def update(self, node: FileNode_Base = None) -> bool:
		# check if connected
		if not self._is_started:
			return False

		# if node is not given
		# then root node is selected
		if node is None:
			node = self._root

		# check if node is ready or not
		if not node.is_mounted:
			return False
		# if not self._root.getIsChild(node, is_recursive=True):
		# 	return False

		# actual update
		return self._update_(node)

	# Protected
	def _update_(self, node: FileNode_Base, path: str = "./") -> bool:
		# ----- file -----
		if not node.is_dir:
			return self._updateNode_(node, path)

		# ----- dir -----
		# try to create the directory (if directory already exist, then nothing will be done)
		path += node.name
		self._enterDir_(path)

		# update children
		for child in node.children:
			self._update_(child, path)

	def _start_(self) -> bool:
		raise NotImplementedError

	def _end_(self) -> bool:
		raise NotImplementedError

	def _enterDir_(self, path: str) -> bool:
		raise NotImplementedError

	def _updateNode_(self, node: FileNode_Base, path: str) -> bool:
		raise NotImplementedError

	# currently the path must be absolute path
	# currently the only valid separator is "/"
	def _getNode_Path_(self, path: str) -> FileNode_Base:
		# CHECK
		# check if path is empty or not
		# path is not allowed to be empty
		if not path:
			return None

		# CONFIG
		path_element_list: List[str] = path.split("/")

		# starting from root
		node_cur: FileNode_Base = self._root

		# CORE
		for element in path_element_list:

			# special case
			# - same level
			# - previous level (if available)
			if element == ".":
				continue

			if element == "..":
				if node_cur.parent is None:  # reaching the root
					return None
				node_cur = node_cur.parent
				continue

			# search child
			node_next: FileNode_Base = node_cur.getChild_Name(element)
			if node_next is None:
				return None
			if not node_next.is_dir:
				return None

			node_cur = node_next

		# RET
		return node_cur

	# returned path is the path to directory of the node
	def _getPath_Node_(self, node: FileNode_Base) -> str:
		# CHECK
		if node is None:
			return None
		if not node.is_mounted:
			return None

		# CONFIG
		content: 	str 			= ""
		node_cur: 	FileNode_Base = node.parent

		# CORE
		while node_cur is not None:
			content 	= node_cur.name + "/" + content
			node_cur 	= node_cur.parent

		# RET
		return content

	# Operator Overloading / Magic Function
	def __str__(self):
		return self._getString_Node_(self._root, 0, 4)

	def _getString_Node_(self, node: FileNode_Base, level: int, indent: int) -> str:
		# self
		content: str = ""
		content += ' ' * level * indent
		content += node.name
		content += "\n"

		# child
		for child in node.children:
			content += self._getString_Node_(child, level + 1, indent)

		return content


class FileControl_Local(FileControl_Base):

	def __init__(self):
		super().__init__()

		# data
		self._path_local_root: str = ""

		# operation
		# ...

	def __del__(self):
		return

	# Property
	@property
	def path_local_root(self):
		return self._path_local_root

	# Operation
	def setLocalRoot(self, path: str) -> bool:
		if not os.path.isdir(path):
			return False

		self._path_local_root = path
		return True

	# Protected
	def _start_(self) -> bool:
		if self._path_local_root == "":
			return False

		# mark root as mounted
		self._root.is_mounted = True

		return True

	def _end_(self) -> bool:
		self._root.is_mounted = False
		return True

	def _enterDir_(self, path: str) -> bool:
		# check if the path exist or not
		# if exist then nothing is needed to do
		path_local: str = os.path.join(self._path_local_root)
		if os.path.isdir(path_local):
			return True

		# path not exist, then need to create the directory
		# but first need to check if the name is already used by a file
		# name collision is default no allowed
		pathlib.Path(path_local).mkdir()

	def _updateNode_(self, node: FileNode_Base, path: str) -> bool:
		# filename
		# no change to filename
		#
		# format:
		# if extension exist:	node.extension
		# else:					node
		if not node.extension:
			filename = node.name
		else:
			filename = node.name + "." + node.extension

		# path
		path = os.path.join(self._path_local_root, path)

		# load, dump
		if node.is_require_load:
			return node.load(path, filename)

		if node.is_require_dump:
			return node.dump(path, filename)

		# no operation is required
		return True
