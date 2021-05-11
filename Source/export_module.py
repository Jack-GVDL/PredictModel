"""
It is the reverse action of 'import_module.py'
"""


import os
import json
import shutil


# path to current file
temp 			= __file__.split("\\")
path_current 	= "/".join(temp)

temp 			= path_current.split("/")
path_current 	= "/".join(temp[:-1])


# assumed: ImportConfig.json must exist
# get data from ImportConfig
with open(os.path.join(path_current, "ExportConfig.json"), "r") as f:
	data = json.load(f)


# get path to target (location that store the module)
path_dir_repo 	= data["Path_Src"]
path_dir_cur	= data["Path_Dst"]
module_list		= data["Module"]


# remove package in current directory (if exist), then
# copy package from the target path
for module in module_list:
	path_cur 	= os.path.join(path_current, path_dir_cur, module)
	path_repo 	= os.path.join(path_current, path_dir_repo, module)

	# check if target directory existed or not
	if not os.path.isdir(path_cur):
		print(f"Package not existed: {path_cur}")
		continue

	if os.path.isdir(path_repo):
		shutil.rmtree(path_repo, ignore_errors=True)
		print(f"Removed: {path_repo}")

	shutil.copytree(path_cur, path_repo)  # copy: path_cur (current) -> path_repo (repo)
	print(f"Copied: src: {path_cur}, dst: {path_repo}")
