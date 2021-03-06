# Reference
# https://github.com/ndrplz/ConvLSTM_pytorch


import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

	def __init__(self, input_dim, hidden_dim, kernel_size, bias):

		super(ConvLSTMCell, self).__init__()

		self.input_dim	= input_dim
		self.hidden_dim = hidden_dim

		self.kernel_size	= kernel_size
		self.padding		= kernel_size[0] // 2, kernel_size[1] // 2
		self.bias			= bias

		self.conv = nn.Conv2d(
			in_channels		= self.input_dim + self.hidden_dim,
			out_channels	= 4 * self.hidden_dim,  # 4 copy (3 sigmoid and 1 tanh, refer to LSTM)
			kernel_size		= self.kernel_size,
			padding			= self.padding,
			bias			= self.bias)

	def forward(self, input_tensor, cur_state):
		h_cur, c_cur = cur_state

		combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

		combined_conv = self.conv(combined)

		cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
		i = torch.sigmoid(cc_i)
		f = torch.sigmoid(cc_f)
		o = torch.sigmoid(cc_o)
		g = torch.tanh(cc_g)

		c_next = f * c_cur + i * g
		h_next = o * torch.tanh(c_next)

		return h_next, c_next

	def init_hidden(self, batch_size, image_size):
		height, width = image_size
		return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
				torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM2d_2(nn.Module):

	def __init__(
		self,
		dim, kernel_size, size_time,
		bias=True, return_all_layers=True
	):

		super().__init__()

		self.input_dim		= dim
		self.hidden_dim		= dim
		self.kernel_size 	= kernel_size
		self.bias 			= bias
		self.size_time		= size_time
		self.cell_list		= nn.ModuleList()

		# operation
		self.cell_list.append(ConvLSTMCell(
			input_dim=self.input_dim,
			hidden_dim=self.hidden_dim,
			kernel_size=self.kernel_size,
			bias=bias
		))

	def forward(self, input_tensor: torch.Tensor):
		# input_tensor: shape: [B, C, H, W]
		b, _, _, h, w = input_tensor.size()

		# init hidden, cell state
		c = torch.zeros(b, self.hidden_dim, h, w, device=input_tensor.device)
		h = input_tensor

		# time looping
		output_list = []
		state_list	= []

		for t in range(self.size_time):

			# call "forward" function in ConvLSTMCell
			h, c = self.cell_list[0](
				input_tensor=h,
				cur_state=[h, c]
			)

			# record
			output_list.append(h)
			state_list.append(c)

		# RET
		return output_list, state_list
