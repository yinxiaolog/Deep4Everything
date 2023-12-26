import torch
from d2l_ai import torch as d2l
from torch import nn

from utils.constant import *


class Mlm(nn.Module):
    def __init__(self, vocab_size, num_inputs, hidden):
        super(Mlm, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, vocab_size)
        )

    def forward(self, x: torch.Tensor, positions):
        num_positions = positions.shape[1]
        positions = positions.reshape(-1)
        batch_size = x.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_positions)
        masked_x = x[batch_idx, positions]
        masked_x = masked_x.reshape((batch_size, num_positions, -1))
        mlm_y_hat = self.mlp(masked_x)
        return mlm_y_hat


class Nsp(nn.Module):
    def __init__(self, inputs, hidden):
        super(Nsp, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class BertEncoder(nn.Module):
    def __init__(self, net_parameters):
        super(BertEncoder, self).__init__()
        self.vocab_size = net_parameters[VOCAB_SIZE]
        self.embedding_dim = net_parameters[EMBEDDING_DIM]
        self.layers = net_parameters[LAYERS]
        self.norm_shape = [net_parameters[EMBEDDING_DIM]]
        self.ffn_input = net_parameters[EMBEDDING_DIM]
        self.ffn_hidden = net_parameters[FFN_HIDDEN]
        self.heads = net_parameters[HEADS]
        self.dropout = net_parameters[DROPOUT]
        self.max_len = net_parameters[MAX_LEN]
        self.key_size = net_parameters[EMBEDDING_DIM]
        self.query_size = net_parameters[EMBEDDING_DIM]
        self.value_size = net_parameters[EMBEDDING_DIM]
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.segment_embedding = nn.Embedding(2, self.embedding_dim)
        self.seq = nn.Sequential()
        for i in range(self.layers):
            self.seq.add_module(f'{i}', d2l.EncoderBlock(
                self.key_size, self.query_size, self.value_size,
                self.embedding_dim, self.norm_shape, self.ffn_input, self.ffn_hidden,
                self.heads, self.dropout, True
            ))
        self.position_embedding = nn.Parameter(torch.randn(1, self.max_len, self.embedding_dim))

    def forward(self, tokens, segments, valid_lens):
        x = self.token_embedding(tokens) + self.segment_embedding(segments)
        x = x + self.position_embedding.data[:, :x.shape[1], :]
        for block in self.seq:
            x = block(x, valid_lens)
        return x
