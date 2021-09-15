from torchvision import datasets, transforms

import argparse
import torch

def infer():
	# Save predictions and the ground truth data in lists
	predictions = []
	groundtruth = []
	with torch.no_grad():
		for data, target in infer_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)

			# get the index of max log-probability
			pred = output.max(1, keepdim=True)[1].squeeze(1)
			
			predictions.append(pred)
			groundtruth.append(target)

	# Concatenate lists to torch tensors
	predictions = torch.cat(predictions)
	groundtruth = torch.cat(groundtruth)

	# Concatenate tensors before saving
	out = torch.cat((predictions.unsqueeze(1), groundtruth.unsqueeze(1)), dim=1)

	# Save the output tensor
	torch.save(out, opt.chkpt.replace(".pt", ".tst"))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--chkpt', type=str, help="model file", required=True)
	opt = parser.parse_args()

	# If GPU exists, run the computations with CUDA accelerator
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Inference dataset
	infer_loader = torch.utils.data.DataLoader(datasets.MNIST(root='.', train=False, transform=transforms.Compose([transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))])), batch_size=64, shuffle=True, num_workers=4)

	# Load model from the given checkpoint
	model_fname = opt.chkpt.replace(".pt", "")
	import_func = "from " + model_fname + " import Net"
	exec(import_func)

	# Initiate model and update the state dict to the latest state
	model = Net().to(device)
	model.load_state_dict(torch.load(opt.chkpt))

	infer()
