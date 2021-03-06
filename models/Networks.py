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
        # bring shape [bs, depth, height, widt, rgb+a]
        # to shape [bs, rgb+a, depth, height, width]
        
        x = x.permute(0, 4, 1, 2, 3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # TODO: do we want to add softmax in forward pass
        return torch.sigmoid(x)

class TextEncoder(nn.Module):
    def __init__(self, vocabulary_size):
        super(TextEncoder, self).__init__()
        """
        Network for shape encoder
        Conv1d -->  [batch_size, in_channels, signal_length]
                    [bs, embeddings, 96]
        
        formula: [(W−K+2P)/S]+1
            W = input
            K = kernel
            P = padding
            S = stride
        """
        self.emb = nn.Embedding(vocabulary_size, 128)
        # define layers of a convolutional neural network
        self.conv1 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(nn.Conv1d(128, 128, 3, stride=1, padding=1),
                                   nn.BatchNorm1d(128),)
        self.conv3 = nn.Conv1d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, 3, stride=1, padding=1),
                                   nn.BatchNorm1d(256))
        self.gru = nn.GRU(input_size=256, hidden_size=256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        des_length = self.compute_description_length(x)

        x = self.emb(x)         # [bs, seq, emb]
        x = x.transpose(1, 2)    # [bs, emb, seq]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # [bs, emb, seq] to [seq, bs , emb) for GRU
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        x, hidden = self.gru(x, None)
        x = F.relu(x)

        x = self.extract_relevant(x, des_length)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.sigmoid(x)

    def compute_description_length(self, batch):
        """
        batch:          [bs, seq]
        des_length:     size of bs containing the length of
                        each description in batch 
        """
        des_length = torch.gt(batch, 0).sum(dim=1).long()
        return des_length

    def extract_relevant(self, out, des_length):
        #TODO: not quite sure what happens here 
        # combination of orignal repo and
        # https://github.com/eriche2016/text2shape.pytorch/blob/master/models/lba_models.py
        
        bs = out.shape[1]
        max_length = out.shape[0]
        out_size = int(out.shape[2])

        masks = (des_length-1).unsqueeze(0).unsqueeze(2).expand(max_length, bs, out_size)
        relevant = out.gather(0, masks)[0]

        return relevant
