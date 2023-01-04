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

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.BOARD_HEIGHT = GameEngine.HEIGHT
        self.BOARD_WIDTH = GameEngine.WIDTH
        NUM_CHANNELS = 512
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, NUM_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(NUM_CHANNELS),
            nn.LeakyReLU(),
            nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(NUM_CHANNELS),
            nn.LeakyReLU(),
            nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(NUM_CHANNELS),
            nn.LeakyReLU(),
            nn.Conv2d(NUM_CHANNELS, NUM_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(NUM_CHANNELS),
            nn.LeakyReLU()
        )
        
        self.fc_common = nn.Sequential(
            nn.Linear(NUM_CHANNELS * self.BOARD_HEIGHT * self.BOARD_WIDTH, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU()
        )
        self.fc_policy = nn.Sequential(
            nn.Linear(512, self.BOARD_WIDTH),
            nn.LogSoftmax(dim=1)
        )
        self.fc_value = nn.Sequential(
            nn.Linear(512, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # input: batch_size x 2 x board_height x board_width
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        batch_size = len(x)
        
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc_common(x)
        
        policy = self.fc_policy(x)
        value = self.fc_value(x)
        
        return policy, value          
                