# The code is adopted from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

import torch.nn as nn
import torch
import torch.nn.functional as F

class Focus(nn.Module):
	# Focus wh information into c-space
	def __init__(self, c1, c2, k=4):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(c1*4, c2, k),
			nn.SiLU(True))

	def forward(self, x):
		x = self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
		return x

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

		# Spatial transformer localization-network
		self.localization = nn.Sequential(
			Focus(1, 8),
			nn.Conv2d(8, 10, kernel_size=5),
			nn.MaxPool2d(2, stride=2),
			nn.SiLU(True)
		)

		# Regressor for the 3 * 2 affine matrix
		self.fc_loc = nn.Sequential(
			nn.Linear(10 * 3  * 3, 32),
			nn.SiLU(True),
			nn.Linear(32, 3 * 2)
		)

		# Initialize the weights/bias with identity transformation
		self.fc_loc[2].weight.data.zero_()
		self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

	# Spatial transformer network forward function
	def stn(self, x):
		xs = self.localization(x)
		xs = xs.view(-1, 10 * 3 * 3)
		theta = self.fc_loc(xs)
		theta = theta.view(-1, 2, 3)

		grid = F.affine_grid(theta, x.size())
		x = F.grid_sample(x, grid)

		return x

	def forward(self, x):
		# transform the input
		x = self.stn(x)

		# Perform the usual forward pass
		x = F.silu(F.max_pool2d(self.conv1(x), 2))
		x = F.silu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(-1, 320)
		x = F.silu(self.fc1(x))
		x = self.fc2(x)

		return F.log_softmax(x, dim=1)

