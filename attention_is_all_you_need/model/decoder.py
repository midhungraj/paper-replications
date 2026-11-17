import torch.nn as nn

from attention import MultiHeadAttension
from feed_forward import PositionWiseFeedForaward
from layer_norm import LayerNorm

from transformer_embedding import TransformerEmbedding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.self_attension = MultiHeadAttension(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.drop_out1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attension = MultiHeadAttension(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.drop_out2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionWiseFeedForaward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.drop_out3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        _x = dec
        x = self.self_attension(Q=dec, K=dec, V=dec, mask=trg_mask)

        x = self.drop_out1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attension(Q=x, K=enc, V=enc)

            x = self.drop_out2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.drop_out3(x)
        x = self.norm3(x + _x)

        return x


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                                for _ in range(n_layers)]
                                                )

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        output = self.linear(trg)
        return output

