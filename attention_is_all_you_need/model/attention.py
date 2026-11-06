import math
import torch
import torch.nn as nn


class ScaledDotProductAttension(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        # Calculate similarity scores
        scores = Q @ K.transpose(-2, -1)

        # Scale
        d_k = Q.size(-1)
        scores = scores / math.sqrt(
            d_k
        )  # torch.sqrt() also works fine!, i just using math.sqrt....]

        if mask is not None:
            scores = scores.masked_filled(mask, float("-inf"))

        weights = torch.softmax(scores, dim=1)

        output = weights @ V

        return output, weights


class MultiHeadAttension(nn.Module):
    def __inti__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.attension = ScaledDotProductAttension()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        q, k, v = self.w_q(Q), self.w_k(K), self.w_v(V)

        q, k, v = self.split(q), self.split(k), self.split(v)

        out, attention = self.attension(q, k, v)

        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head

        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = d_tensor * head

        # out = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        out = tensor.transpose(1, 2).reshape(batch_size, length, d_model)
        return out
