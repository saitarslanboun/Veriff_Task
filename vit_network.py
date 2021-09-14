# The code is adopted from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
	def __init__(self, c):
		super().__init__()
		self.q = nn.Linear(c, c, bias=False)
		self.k = nn.Linear(c, c, bias=False)
		self.v = nn.Linear(c, c, bias=False)
		self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=1)

	def forward(self, x):
		x = x.view(x.shape[0], x.shape[1], -1).transpose(1, 2)
		x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
		return x

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.vit = MultiHeadAttention(20)
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):

		# Perform the usual forward pass
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = self.vit(x)
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)

		return F.log_softmax(x, dim=1)

