import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)

        two_i = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(two_i * (-math.log(10000.0) / d_model))

        encoding = torch.zeros(max_len, d_model)

        # Even
        encoding[:, 0::2] = torch.sin(position * div_term)

        # Odd
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])

        # Add batch size
        encoding = encoding.unsqueeze(0)

        # Save as a buffer
        self.register_buffer("encoding", encoding)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor):
        tensor = tensor + self.encoding[:, : tensor.size(1), :]
        return self.dropout(tensor)
