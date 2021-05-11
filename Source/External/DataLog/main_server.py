from typing import *
import json
import os
from Source import *


# Function
def main() -> None:
	# ----- Config -----
	# read config
	with open("./Config_Server.json", "r") as f:
		data = json.load(f)

	# get path
	path_data = data["Path_Data"]

	# ----- Control_Data -----
	# create Control_Data
	control_data = Control_Data()

	# try to read data from json
	try:
		with open(path_data, "r") as f:
			data = json.load(f)
			control_data.setDictData(data)
	except Exception:
		pass

	# ----- Server -----
	# start server
	Server_main(control_data, path_data)


# Operation
if __name__ != "__main__":
	raise RuntimeError


main()
