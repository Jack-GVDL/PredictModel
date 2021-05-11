from typing import *
from .Log_Data import Log_Data, Control_Data
import json
from flask import Flask, request
from flask_cors import CORS
import logging


# Data
control_data_: 	List[Control_Data]	= []
path_save_:		List[str]			= ["./Data/Data_Temp.json"]

app = Flask(__name__)
CORS(app)

log = logging.getLogger("werkzeug")
log.disabled = True


# Function
def Server_main(control_data: Control_Data, path_save: str = None) -> None:
	# Control_Log
	control_data_.append(control_data)

	# path_save
	if path_save is not None:
		path_save_[0] = path_save

	# run
	app.run(port=8001)


def getArgument(target: List[Any], name: str, func_convert: Callable[[str], Any]) -> bool:
	try:
		data = request.args.get(name)
		data = func_convert(data)

	except Exception:
		return False

	# if there is existing item
	# it is assumed that that item is the default value
	if not target:
		target.append(data)
	else:
		target[0] = data
	return True


def convertData(data_list: List[Any], type_: int) -> bool:
	# get convert function
	convert_func_table: Dict = {
		Log_Data.DataType.NONE:		None,
		Log_Data.DataType.BOOL:		lambda x: bool(x),
		Log_Data.DataType.INT:		lambda x: int(x),
		Log_Data.DataType.FLOAT:	lambda x: float(x),
		Log_Data.DataType.STR:		lambda x: str(x)
	}

	if type_ not in convert_func_table:
		return False

	convert_func = convert_func_table[type_]
	if convert_func is None:
		return False

	# convert
	try:
		for i in range(len(data_list)):
			data_list[i] = convert_func(data_list[i])
	except Exception:
		return False

	return True


# ----- route -----
@app.route("/GetList_LogData_Name")
def getList_LogData_DataName():
	# ----- compute return data -----
	data_list: List[Any] = []

	# CHECK
	# check if control_data existed or not
	if not control_data_:
		result = json.dumps(data_list)
		return result
	control_data: Control_Data = control_data_[0]

	# [id_, name] to dict
	for log_data in control_data.log_data_list:
		data = [log_data.id_, log_data.name]
		data_list.append(data)

	# to json
	result = json.dumps(data_list)

	# log
	print(f"GetLogList_DataName: size_data_list: {len(data_list)}")

	# RET
	return result


# return a list of id that the data inside is changed
@app.route("/GetList_LogData_Changed")
def getList_LogData_Changed():
	# ----- compute return data -----
	data_list: List[Any] = []

	# CHECK
	# check if control_data existed or not
	if not control_data_:
		result = json.dumps(data_list)
		return result
	control_data: Control_Data = control_data_[0]

	# [id_] to dict
	for id_log_data in control_data.change_list:
		data = id_log_data
		data_list.append(data)

	# to json
	result = json.dumps(data_list)

	# log
	print(f"GetList_LogData_Changed: size_change_list: {len(data_list)}")

	# RET
	return result


@app.route("/AddLog_Data", methods=["POST"])
def addLog_Data():
	# CHECK
	# check if control_data existed or not
	if not control_data_:
		return "{}"
	control_data: 	Control_Data	= control_data_[0]
	path_save:		str				= path_save_[0]

	# ----- get data -----
	# CONFIG
	name:		Any = []
	data_list:	Any = []
	data_type:	Any = []

	# necessary
	if not getArgument(data_type, "type", lambda x: int(x)) or \
		not getArgument(data_list, "data", lambda x: json.loads(x)):
		return "{}"

	# optional
	getArgument(name, "name", lambda x: str(x))

	# unpack
	name:		str 		= name[0]
	data_list:	List[Any]	= data_list[0]
	data_type:	int			= data_type[0]

	# convert type of item in data_list to  corresponding data type
	if not convertData(data_list, data_type):
		return "{}"

	# ----- add data -----
	if not control_data.addLog_Data(data_list, data_type, name):
		return "{}"

	# ----- save data -----
	with open(path_save, "w") as f:
		data = json.dumps(control_data.getDictData(), indent=None, separators=(',', ':'))
		f.write(data)

	# ----- log -----
	print(f"AddLog_Data: size data_list: {len(data_list)}; type: {data_type}; name: {name}")

	return "{}"


@app.route("/RmLog_Data", methods=["POST"])
def rmLog_Data():
	# CHECK
	# check if control_data existed or not
	if not control_data_:
		return "{}"
	control_data: 	Control_Data	= control_data_[0]
	path_save:		str				= path_save_[0]

	# ----- get data -----
	# config
	id_:	Any = []

	# necessary
	if not getArgument(id_, "id", lambda x: int(x)):
		return "{}"

	# unpack
	id_:	int = id_[0]

	# ----- rm data -----
	if not control_data.rmLog_Data(id_):
		return "{}"

	# ----- save data -----
	with open(path_save, "w") as f:
		data = json.dumps(control_data.getDictData(), indent=None, separators=(',', ':'))
		f.write(data)

	# ----- get log -----
	print(f"RmLog_Data: id: {id_}")

	return "{}"


@app.route("/GetLog_Data", methods=["POST"])
def getLog_Data():
	# CHECK
	# check if control_data existed or not
	if not control_data_:
		return "{}"
	control_data: 	Control_Data	= control_data_[0]

	# ----- get data -----
	# config
	id_:	Any = []

	# necessary
	if not getArgument(id_, "id", lambda x: int(x)):
		return "{}"

	# unpack
	id_:	int = id_[0]

	# ----- get data -----
	log_data: Log_Data = control_data.getLog_Data(id_)
	if log_data is None:
		return "{}"

	# convert Log_Data to dict
	# then from dict to json
	data = log_data.getDictData()
	data = json.dumps(data)

	# ----- get log -----
	print(f"GetLog_Data: id: {id_}")

	# RET
	return data


# Operation
if __name__ == "__main__":
	raise RuntimeError
