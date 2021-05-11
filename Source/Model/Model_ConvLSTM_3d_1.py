# Reference
# https://github.com/ndrplz/ConvLSTM_pytorch


from typing import *
import torch.nn as nn
import torch


class ConvLSTM3d_1_Cell(nn.Module):

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

	def initHidden(self, batch_size, image_size) -> Any:
		d, h, w = image_size
		return (
			torch.zeros(batch_size, self.hidden_dim, d, h, w, device=self.conv.weight.device),
			torch.zeros(batch_size, self.hidden_dim, d, h, w, device=self.conv.weight.device)
		)


class ConvLSTM3d_1(nn.Module):

	# Static
	@staticmethod
	def _checkKernelSizeConsistency_(kernel_size):
		if \
			not (isinstance(kernel_size, tuple) or
			(isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
			raise ValueError('`kernel_size` must be tuple or list of tuples')

	@staticmethod
	def _extendForMultiLayer_(param, size_layer):
		if not isinstance(param, list):
			param = [param] * size_layer
		return param

	def __init__(
		self,
		input_dim, hidden_dim, kernel_size, size_layer,
		batch_first=True, bias=True, return_all_layer=False
	):
		super().__init__()

		# ----- data -----
		self._checkKernelSizeConsistency_(kernel_size)

		# make sure that both kernel_size and hidden_dim are lists having len == size_layer
		kernel_size	= self._extendForMultiLayer_(kernel_size, size_layer)
		hidden_dim	= self._extendForMultiLayer_(hidden_dim, size_layer)
		if not len(kernel_size) == len(hidden_dim) == size_layer:
			raise ValueError("Inconsistent list length")

		self.input_dim			= input_dim
		self.hidden_dim			= hidden_dim
		self.kernel_size		= kernel_size
		self.size_layer			= size_layer
		self.batch_first		= batch_first
		self.bias				= bias
		self.return_all_layer 	= return_all_layer
		self.cell_list			= None

		# ----- operation -----
		cell_list = []
		for i in range(0, self.size_layer):
			cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

			cell_list.append(
				ConvLSTM3d_1_Cell(
					input_dim=cur_input_dim,
					hidden_dim=self.hidden_dim[i],
					kernel_size=self.kernel_size[i],
					bias=self.bias))

		self.cell_list = nn.ModuleList(cell_list)

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	def forward(self, input_tensor, hidden_state=None) -> Any:
		if not self.batch_first:
			# [T, B, C, D, H, W] -> [B, T, C, D, H, W]
			input_tensor = input_tensor.permute(1, 0, 2, 3, 4, 5)

		b, _, _, d, h, w = input_tensor.size()

		# implement stateful ConvLSTM
		# now hidden state should be None
		if hidden_state is not None:
			raise NotImplementedError
		else:
			# since the init is done in forward
			# we can set image size here
			hidden_state = self._initHidden_(batch_size=b, image_size=(d, h, w))

		layer_output_list	= []
		last_state_list		= []

		size_time 		= input_tensor.size(1)  # t
		cur_layer_input	= input_tensor

		for index_layer in range(self.size_layer):

			h, c = hidden_state[index_layer]
			output_inner = []

			# foreach time-slice
			for t in range(size_time):

				# call forward() function in ConvLSTM3d_Cell
				h, c, = self.cell_list[index_layer](
					input_tensor=cur_layer_input[:, t, :, :, :, :],
					cur_state=[h, c]
				)
				output_inner.append(h)

			layer_output 	= torch.stack(output_inner, dim=1)
			cur_layer_input	= layer_output

			layer_output_list.append(layer_output)
			last_state_list.append([h, c])

		# check if need to return all layer
		if not self.return_all_layer:
			layer_output_list	= layer_output_list[-1:]
			last_state_list		= last_state_list[-1:]

		return layer_output_list, last_state_list

	# Protected
	def _initHidden_(self, batch_size, image_size) -> Any:
		init_state = []
		for i in range(self.size_layer):
			init_state.append(self.cell_list[i].initHidden(batch_size, image_size))
		return init_state
