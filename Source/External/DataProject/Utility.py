from typing import *
from ..DataChain import *
from ..Util_Cmd import *
from .Handler_DataUpdateLog import Handler_DataUpdateLog


# Function
# update data chain with log
def updateDataChain(data_control: DataControl, target: DataList, is_verbose=True) -> None:
	# log
	log_list: List[DataUpdateLog] = []

	# update
	data_control.update(target, log_list=log_list)

	# verbose
	if is_verbose:

		# printer
		control_content = Control_StringContent()
		handler_log		= Handler_DataUpdateLog()

		handler_log.log_list		= log_list
		handler_log.control_content	= control_content
		handler_log.convert()

		# print
		printer_content = Printer_StringContent()
		printer_content.control_content = control_content

		printer_content.print()
