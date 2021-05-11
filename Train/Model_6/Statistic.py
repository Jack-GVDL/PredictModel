from typing import *
import torch
from torchsummary import summary
import torch.nn as nn
from Source import *
from .Dataset_Train import Dataset_Train


def countParameter(model: nn.Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def Statistic_main() -> None:
	model = Model_6()

	# count parameter
	print(f"Model parameter: {countParameter(model)}")

	# show layer
	# summary(model, input_size=())
