from torch.nn import nn
from torch.nn.functional import F
import torch

class TextEncoder(nn.Module):
    def __init__(self, history_length=1, n_classes=3): 
    super(TextEncoder, self).__init__()
    """
    formula: [(W−K+2P)/S]+1
    W = input
    K = kernel
    P = padding
    S = stride
    """

    # define layers of a convolutional neural network
    self.conv1 = nn.Conv1d(in_channels=16, out_channels=128, kernel_size=3)
    self.conv2 = nn.Conv2d(128, 128, 3)
    self.conv3 = nn.Conv2d(128, 256, 3)
    self.conv4 = nn.Conv2d(256, 256, 3)

    self.fc5 = nn.Linear(self.flat_features, 100)
    self.fc6 = nn.Linear(100, 50)