from torch.nn import nn
from torch.nn.functional import F
import torch

class TextEncoder(nn.Module):
    def __init__(self, vocabulary_size): 
        super(TextEncoder, self).__init__()
        """
        formula: [(W−K+2P)/S]+1
        W = input
        K = kernel
        P = padding
        S = stride
        """

        # define layers of a convolutional neural network
        self.conv1 = nn.Conv1d(vocabulary_size, 128, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(128, 128, 3, stride=1)
        self.conv3 = nn.Conv1d(128, 256, 3, stride=1)
        self.conv4 = nn.Conv1d(256, 256, 3, stride=1)
        self.gru = nn.GRU(input_size=256, hidden_size=256)

        self.fc1 = nn.Linear(self.flat_features, 100)
        self.fc2 = nn.Linear(100, 50)
    
    def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))

		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		x = self.fc2(x)