# The code is adopted from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.fft

class FNet(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		x = x.view(x.shape[0], x.shape[1], -1).transpose(1, 2)
		x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real + x
		return x

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.fnet = FNet()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):

		# Perform the usual forward pass
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = self.fnet(x)
		x = x.reshape(-1, 320)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)

		return F.log_softmax(x, dim=1)

