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
with open(os.path.join(path_current, "ImportConfig.json"), "r") as f:
	data = json.load(f)


# get path to target (location that store the module)
path_dir_src 	= data["Path_Src"]
path_dir_dst	= data["Path_Dst"]
module_list		= data["Module"]


# remove package in current directory (if exist), then
# copy package from the target path
for module in module_list:
	path_dst = os.path.join(path_current, path_dir_dst, module)
	path_src = os.path.join(path_current, path_dir_src, module)

	# check if target directory existed or not
	if not os.path.isdir(path_src):
		print(f"Package not existed: {path_src}")
		continue

	if os.path.isdir(path_dst):
		shutil.rmtree(path_dst, ignore_errors=True)
		print(f"Removed: {path_dst}")

	shutil.copytree(path_src, path_dst)  # copy: path_src (repo) -> path_dst (current)
	print(f"Copied: src: {path_src}, dst: {path_dst}")
