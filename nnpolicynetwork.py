import torch
import torch.nn as nn

class NNPolicyNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(NNPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
