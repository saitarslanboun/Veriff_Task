import torch.nn.modules.conv as conv
import torch.nn as nn
import torch

class AddCoords(nn.Module):
	def __init__(self):
		super(AddCoords, self).__init__()

	def forward(self, x):
		batch_size_shape, channel_in_shape, dim_j, dim_i = x.shape
		ii_ones = torch.ones([1, 1, 1, dim_i]).to(x.device)
		jj_ones = torch.ones([1, 1, 1, dim_j]).to(x.device)

		ii_range = torch.arange(dim_j).to(x.device).float()
		jj_range = torch.arange(dim_i).to(x.device).float()
		ii_range = ii_range[None, None, :, None]
		jj_range = jj_range[None, None, :, None]

		ii_channel = torch.matmul(ii_range, ii_ones).long()
		jj_channel = torch.matmul(jj_range, jj_ones).long()

		# transpose j
		jj_channel = jj_channel.permute(0, 1, 3, 2)

		ii_channel = ii_channel.float() / (dim_j - 1)
		jj_channel = jj_channel.float() / (dim_i - 1)

		ii_channel = ii_channel * 2 - 1
		jj_channel = jj_channel * 2 - 1

		ii_channel = ii_channel.repeat(batch_size_shape, 1, 1, 1)
		jj_channel = jj_channel.repeat(batch_size_shape, 1, 1, 1)

		out = torch.cat([x, ii_channel, jj_channel], dim=1)

		return out
		


class CoordConv2d(conv.Conv2d):
	def __init__(self, c1, c2, kernel_size):
		super(CoordConv2d, self).__init__(c1, c2, kernel_size)

		self.addcoords = AddCoords()
		self.conv = nn.Conv2d(c1 + 2, c2, kernel_size)

	def forward(self, x):
		x = self.addcoords(x)
		x = self.conv(x)

		return x
