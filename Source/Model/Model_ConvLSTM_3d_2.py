# Reference
# https://github.com/ndrplz/ConvLSTM_pytorch


from typing import *
import torch.nn as nn
import torch


class ConvLSTM3d_2_Cell(nn.Module):

	def __init__(self, input_dim, hidden_dim, kernel_size, bias):
		super().__init__()

		# data
		self.input_dim		= input_dim
		self.hidden_dim		= hidden_dim

		self.kernel_size	= kernel_size
		self.padding		= [kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2]
		self.bias			= bias

		# Conv3d
		# reference
		# - https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
		#
		# input size:	[N, C, D, H, W]
		# output size:	[N, C, D, H, W]
		self.conv = nn.Conv3d(
			in_channels		= self.input_dim + self.hidden_dim,
			out_channels	= 4 * self.hidden_dim,  # 4 copy (3 sigmoid, 1 tanh, refer to LSTM)
			kernel_size		= self.kernel_size,
			padding			= self.padding,
			bias			= self.bias
		)

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	def forward(self, input_tensor, cur_state) -> Any:
		# input_tensor: [N, C, D, H, W]
		h_cur, c_cur 	= cur_state
		combined 		= torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

		# conv3d
		combined_conv	= self.conv(combined)

		cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
		i = torch.sigmoid(cc_i)
		f = torch.sigmoid(cc_f)
		o = torch.sigmoid(cc_o)
		g = torch.tanh(cc_g)

		c_next = f * c_cur + i * g
		h_next = o * torch.tanh(c_next)

		return h_next, c_next


class ConvLSTM3d_2(nn.Module):

	def __init__(
		self,
		dim, kernel_size,
		size_layer=1, bias=True
	):

		super().__init__()

		# data
		self.input_dim			= dim
		self.hidden_dim			= dim
		self.kernel_size		= kernel_size
		self.bias				= bias

		self.size_layer			= size_layer
		self.cell_list			= nn.ModuleList()

		# operation
		for _ in range(size_layer):
			self.cell_list.append(ConvLSTM3d_2_Cell(
				input_dim=self.input_dim,
				hidden_dim=self.hidden_dim,
				kernel_size=self.kernel_size,
				bias=bias
			))

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	# mask_list
	# True:		has input tensor
	# False: 	do not have input tensor
	def forward(self, input_tensor: torch.Tensor, mask_list: List[bool]) -> Any:
		# assumed
		# - mask_list is not empty
		# - first item in mask_list must be True
		# - number of True item in mask_list == input_tensor.shape[1] (which is T) (TODO)
		assert mask_list
		assert mask_list[0]

		# input_tensor: shape: [B, T, C, D, H, W]
		b, _, _, d, h, w = input_tensor.size()

		# init hidden state
		c = torch.zeros(b, self.hidden_dim, d, h, w, device=input_tensor.device)
		h = input_tensor[:, 0, :, :, :, :]

		# time looping
		output_list	= []
		state_list	= []

		index: int = 0
		for mask in mask_list:

			# has input tensor
			if mask:
				layer_input = input_tensor[:, index, :, :, :, :]
				index += 1

			# do not have input tensor
			else:
				layer_input = h

			# cell forwarding
			for index_layer in range(self.size_layer):
				h, c = self.cell_list[index_layer](
					input_tensor=layer_input,
					cur_state=[h, c]
				)

			# record
			output_list.append(h)
			state_list.append([h, c])

		# RET
		return output_list, state_list

	# Protected
	# ...
