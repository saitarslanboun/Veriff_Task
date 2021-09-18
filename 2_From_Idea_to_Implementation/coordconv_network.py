# The STN Net code is adopted from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
# CoordConv code is adopted from https://github.com/walsvid/CoordConv

import torch.nn as nn
import torch
import torch.nn.functional as F		
import torch.nn.modules.conv as conv

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

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = CoordConv2d(1, 10, kernel_size=5)
		self.conv2 = CoordConv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

		# Spatial transformer localization-network
		self.localization = nn.Sequential(
			CoordConv2d(1, 8, kernel_size=7),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True),
			CoordConv2d(8, 10, kernel_size=5),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True)
		)

		# Regressor for the 3 * 2 affine matrix
		self.fc_loc = nn.Sequential(
			nn.Linear(10 * 3  * 3, 32),
			nn.ReLU(True),
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
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)

		return F.log_softmax(x, dim=1)

