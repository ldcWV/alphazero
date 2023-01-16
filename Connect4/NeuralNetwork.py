import os
import torch
from torch import nn
from Connect4.Game import GameEngine
from torch.utils.data import Dataset, DataLoader

class Connect4Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        state, policy, value = self.data[idx]
        state = state.toNeuralNetworkInput()
        policy = torch.Tensor(policy)
        value = torch.tensor(value)
        return state, policy, value


# based on https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out
        
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.BOARD_HEIGHT = GameEngine.HEIGHT
        self.BOARD_WIDTH = GameEngine.WIDTH
        NUM_CHANNELS = 128
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, NUM_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(NUM_CHANNELS),
            nn.ReLU()
        )
        
        self.residual1 = ResidualBlock(NUM_CHANNELS, NUM_CHANNELS)
        self.residual2 = ResidualBlock(NUM_CHANNELS, NUM_CHANNELS)
        self.residual3 = ResidualBlock(NUM_CHANNELS, NUM_CHANNELS)
        self.residual4 = ResidualBlock(NUM_CHANNELS, NUM_CHANNELS)
        self.residual5 = ResidualBlock(NUM_CHANNELS, NUM_CHANNELS)
        self.residual6 = ResidualBlock(NUM_CHANNELS, NUM_CHANNELS)
        self.residual7 = ResidualBlock(NUM_CHANNELS, NUM_CHANNELS)
        
        self.fc_common = nn.Sequential(
            nn.Linear(NUM_CHANNELS * self.BOARD_HEIGHT * self.BOARD_WIDTH, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.BOARD_WIDTH),
            nn.LogSoftmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # input: batch_size x 2 x board_height x board_width
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        batch_size = len(x)
        
        x = self.conv(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.residual5(x)
        x = self.residual6(x)
        x = self.residual7(x)
        
        x = x.view(batch_size, -1)
        x = self.fc_common(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value          
                