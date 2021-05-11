from typing import *
from printy import printy


class _PrintyColorTable_:

	# data
	# List[Tuple[flag, color_name]]
	color: List[Tuple[str, str]] = [
		("k", 	"black"),
		("g",	"grey"),
		("w",	"white"),
		("r",	"red"),
		("n",	"green"),
		("y",	"yellow"),
		("c",	"blue"),  # redirect to cyan
		("m>",	"magenta"),
		("c",	"cyan"),
		("o",	"orange"),
		("<p",	"purple")
	]

	# operation
	@classmethod
	def getFlag_Color(cls, color: str) -> str:
		result: List[Tuple[str, str]] = list(filter(lambda x: x[1] == color, cls.color))
		if not result:
			return None
		return result[0][0]


class StringContent:

	def __init__(self, content: str, color_fore: str = None, color_back: str = None) -> None:
		super().__init__()

		# data
		self.content:		str = ""
		self.color_fore:	str = None
		self.color_back:	str = None

		# operation
		self.content	= content
		self.color_fore	= color_fore
		self.color_back	= color_back

	def __del__(self) -> None:
		return

	# Operation
	# ...


class Control_StringContent:

	def __init__(self) -> None:
		super().__init__()

		# data
		self.content_list:	List[StringContent]				= []
		self.func_output:	Callable[[StringContent], None]	= None

		# operation
		# ...

	def __del__(self) -> None:
		return

	# Operation
	def addContent(self, content: StringContent) -> bool:
		self.content_list.append(content)
		return True

	def addString(self, string: str) -> bool:
		self.content_list.append(StringContent(string))
		return True

	# TODO: not yet completed
	def rmContent(self, content: StringContent) -> bool:
		return False

	def output(self) -> bool:
		if self.func_output is None:
			return False

		# output all content (one-by-one)
		for content in self.content_list:
			self.func_output(content)

		# output will clear the output buffer (content_list)
		self.content_list.clear()
		
		return True

	# Operator Overloading
	def __iadd__(self, other):
		# check type
		if type(other) == Control_StringContent:
			self.content_list.extend(other.content_list)
		elif type(other) == StringContent:
			self.addContent(other)
		elif type(other) == str:
			self.addString(other)

		return self


class Printer_StringContent:

	def __init__(self):
		super().__init__()

		# data
		self.control_content: Control_StringContent = None

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def print(self) -> bool:
		# CHECK
		if self.control_content is None:
			return False

		# CORE
		self.control_content.func_output = self._outputScreen_
		self.control_content.output()

		return True

	# Protected

	def _outputScreen_(self, content: StringContent) -> None:
		# termcolor
		# if content.color_fore is None:
		# 	print(content.content, end=' ')
		# else:
		# 	print(colored(content.content, content.color_fore), end=' ')

		# printy
		# get flag
		color_fore: str = None

		if content.color_fore is not None:
			color_fore = _PrintyColorTable_.getFlag_Color(content.color_fore)

		# print content (default)
		if content.color_fore is None:
			print(content.content, end='')
			return

		# special handling of "[" and "]"
		# replace "[" and "]" with "\[" and "\]" respectively
		# string: str = content.content
		# if '[' in string or ']' in string:

		string = ""
		for s in content.content:
			if s == '[':
				string += '\['
			elif s == ']':
				string += '\]'
			else:
				string += s

		# print content (color)
		printy(string, color_fore, end='')
