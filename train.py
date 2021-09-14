# The training code is adopted from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

from torchvision import datasets, transforms

import torch
torch.manual_seed(0)

import random
random.seed(0)

import numpy
numpy.random.seed(0)

import torch.optim as optim
import torch.nn.functional as F
import argparse
import logging

def validate():

	# Move model state to the validation mode
	with torch.no_grad():
		model.eval()
		valid_loss = 0
		correct = 0

		for data, target in valid_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)

			# sum up batch loss
			valid_loss += F.nll_loss(output, target, size_average=False).item()

			# get the index of the max log-probability
			pred = output.max(1, keepdim=True)[1]
			correct += pred.eq(target.view_as(pred)).sum().item()

		# Average validation loss
		valid_loss /= len(valid_loader.dataset)

		# Logging
		logger.info('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(valid_loss, correct, len(valid_loader.dataset),
			100. * correct / len(valid_loader.dataset)))

	return valid_loss

def train(epoch):

	# Activate the model in the training mode
	model.train()

	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)

		# Set gradients to zero before the backpropagation
		optimizer.zero_grad()

		# Forward propagation
		output = model(data)

		#  Negative log likelihood loss
		loss = F.nll_loss(output, target)

		# Backward propagation
		loss.backward()

		# Update the model parameters after the backpropagation
		optimizer.step()

		# Periodical logging
		if batch_idx % 500 == 0:
			logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment", type=str, default="baseline", help="choose experiment type between |baseline|coordconv|vit|fnet|focus|")
	opt = parser.parse_args()

	logging.basicConfig(filename=opt.experiment+"_out.log", filemode="w")
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)

	# If the machine has GPU, use CUDA accelerator for the computations
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Loading the model
	if opt.experiment == 'baseline':
		from baseline_network import Net
	elif opt.experiment == 'coordconv':
		from coordconv_network import Net
	elif opt.experiment == 'vit':
		from vit_network import Net
	elif opt.experiment == 'fnet':
		from fnet_network import Net
	elif opt.experiment == 'focus':
		from focus_network import Net
	model = Net().to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters())
	print (pytorch_total_params)
	exit()

	# Define maximum number of training epochs
	max_num_epochs = 1000

	# Define the maximum number of training epochs for the early stopping
	tolerance_epochs = 20

	# Training dataset
	train_loader = torch.utils.data.DataLoader(datasets.MNIST(root='.', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))])), batch_size=64, shuffle=True, num_workers=4)

	# Validation dataset
	valid_loader = torch.utils.data.DataLoader(datasets.MNIST(root='.', train=False, transform=transforms.Compose([transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))])), batch_size=64, shuffle=True, num_workers=4)

	# Stochastic Gradient Descent optimizer, setting with default hyperparameters
	optimizer = optim.SGD(model.parameters(), lr=0.01)

	# Save validation losses in the list after every epoch
	valid_losses = []

	# Start training
	for epoch in range(max_num_epochs):
		train(epoch)
		valid_loss = validate()
		valid_losses.append(valid_loss)

		# If there is no drop on the validation loss for x epochs, stop training in order to prevent overfitting
		if valid_loss == min(valid_losses):
			tolerance = 0
			state_dict = model.state_dict()
		else:
			tolerance += 1

		if tolerance == tolerance_epochs:
			logger.info("The model has reached convergence. Training is being stopped.")
			break

	logger.info("Saving the model...")
	model_name = opt.experiment + "_network.pt"
	torch.save(state_dict, model_name)
	logger.info("Model is saved.")
