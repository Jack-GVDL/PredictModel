from typing import *
from ..External.DataChain import *
from ..External.Util_Cmd import *


# Data Structure
class Handler_DataUpdateLog:

	def __init__(self):
		super().__init__()

		# data
		self.log_list:			List[DataUpdateLog] = []
		self.control_content:	Control_StringContent = None

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def convert(self) -> bool:
		# CHECK
		if self.control_content is None:
			return False
		if not self.log_list:
			return True

		# CORE
		if not self._convert_():
			return False
		return True

	# Protected
	def _convert_(self) -> bool:
		# CONFIG
		string_list: List[List[str]] = []

		# ----- add title to string_list -----
		string_list.append([
			"src_list",
			"dst",
			"conversion",
			"state",
			"input",
			"output",
			"duration"
		])

		# ----- get string -----
		for log in self.log_list:
			string_list.append(self._convertSingle_(log))

		# ----- get max length -----
		# assumed: log_list must not be empty
		max_length_list: List[int] = [0 for _ in range(len(string_list[0]))]
		for log_string in string_list:
			for i in range(len(max_length_list)):
				max_length_list[i] = max(max_length_list[i], len(log_string[i]))

		# ----- make title -----
		for i in range(len(max_length_list)):
			# add separator
			if i != 0:
				self.control_content += " | "

			# add title
			content: str			= string_list[0][i]
			content					+= ' ' * (max_length_list[i] - len(content))
			self.control_content 	+= content

		self.control_content += "\n"

		# ----- make horizontal separator -----
		for i in range(len(max_length_list)):
			# add intersection
			if i != 0:
				self.control_content += "-*-"

			# add separator
			content: str 			= '-' * max_length_list[i]
			self.control_content 	+= content

		if len(string_list) != 1:
			self.control_content += "\n"

		# ----- generate content -----
		for index_log, log_string in enumerate(string_list):

			# first row is title
			# therefore is handled differently
			if index_log == 0:
				continue

			for i in range(len(max_length_list)):

				# separator
				if i != 0:
					self.control_content += " | "

				# color
				# TODO: find a prettier way to do the same thing
				color: str = None
				if i == 3:
					if log_string[i] == "success":
						color = "green"
					else:
						color = "red"

				# add content
				content: str 	= log_string[i]
				content 		+= ' ' * (max_length_list[i] - len(content))
				self.control_content += StringContent(content, color)

			# newline
			if index_log != len(string_list) - 1:
				self.control_content += '\n'

		return True

	def _convertSingle_(self, log: DataUpdateLog) -> List[str]:
		# item that need to show on screen
		# - src_list name
		# - dst name
		# - conversion name
		# - result / state
		# - data input
		# - data output
		# - duration
		# - miscellaneous (TODO)

		# ----- get string -----
		# src_list name
		src_name: str = ""
		for src in log.src_list:
			if len(src_name) != 0:
				src_name += ", "
			src_name += src.name

		# dst name
		dst_name: str = log.dst.name

		# conversion name
		conversion_name: str = log.conversion.name

		# state
		state: str = "success" if log.state == DataUpdateLog.State.SUCCESS else "fail"

		# data converted
		size_input: 	str = str(log.size_input)
		size_output:	str = str(log.size_output)

		# duration
		duration: str = f"{log.duration:.2f}"

		return [
			src_name,
			dst_name,
			conversion_name,
			state,
			size_input,
			size_output,
			duration
		]
