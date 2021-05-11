import torch


class Util_Torch:

	# Static Function
	@classmethod
	def getString_Environment(cls) -> str:
		# get device
		env_device = "cuda:0" if torch.cuda.is_available() else "cpu"
		env_device = torch.device(env_device)

		# generate content
		content: str = ""
		content += f"CUDA available: {torch.cuda.is_available()}\n"
		content += f"Device name:    {torch.cuda.get_device_name(env_device)}\n"
		content += f"Device memory:  {torch.cuda.get_device_properties(env_device).total_memory}\n"

		return content

	def __init__(self):
		super().__init__()

		# data
		# ...

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	# ...

	# Protected
	# ...

