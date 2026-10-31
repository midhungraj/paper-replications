import torch
import torch.nn as nn

class PositionWiseFeedForaward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        self.l1 = nn.Linear(d_model, hidden)
        self.l2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.l1(tensor)
        tensor = self.dropout(tensor)
        tensor = self.relu(tensor)
        tensor = self.l2(tensor)
        return tensor

