import torch.nn as nn
import torch.nn.functional as F
import torch

class ShapeEncoder(nn.Module):
	"""
	Network for shape encoder
	Conv3d --> 	[batch_size, in_channels (RGB+alpha), depth, height, width]
				[bs, 4, 32, 32, 32]
	"""
	def __init__(self):
		super(ShapeEncoder, self).__init__()
		self.conv1 = nn.Sequential(nn.Conv3d(4, 64,kernel_size=3, stride=2),
								   nn.BatchNorm3d(64))
		self.conv2 = nn.Sequential(nn.Conv3d(64, 128,kernel_size=3, stride=2),
								   nn.BatchNorm3d(128))
		self.conv3 = nn.Sequential(nn.Conv3d(128, 256,kernel_size=3, stride=2),
						           nn.BatchNorm3d(256))
		self.pool = nn.MaxPool3d((2, 2, 2))
		self.fc = nn.Linear(256, 128)


	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = self.pool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		# TODO: do we want to add softmax in forward pass
		return x