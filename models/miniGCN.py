import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, resid_pdrop, attn_pdrop,  n_head, adj):
        super().__init__()
        assert n_embd % n_head == 0
        self.value = nn.Linear(n_embd, n_embd)
        self.adj = adj
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = self.adj.masked_fill(self.adj == 0, float('-inf'))
        att = F.softmax(att.to(x.device), dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, resid_pdrop, attn_pdrop,  n_head, adj):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, resid_pdrop, attn_pdrop,  n_head, adj)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

def create_sinusoidal_embeddings(n_pos, dim, out):
    out.detach_()
    out.requires_grad = False
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))

class GCN(nn.Module):

    def __init__(self, n_embd, n_layer, n_head, block_size=500, attn_pdrop=0.1, embd_pdrop=0.1, resid_pdrop=0.1, adj=None):
        super().__init__()
        # input embedding stem
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.position_enc = create_sinusoidal_embeddings(n_pos=500, dim=n_embd, out=self.pos_emb.weight)
        self.drop = nn.Dropout(embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, resid_pdrop, attn_pdrop,  n_head, adj) for _ in range(n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.block_size = block_size
        self.apply(self._init_weights)

        logger.info("number of parameters in transformer layers: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        b, seq_len, seq_dim = x.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_embeddings = self.pos_emb(position_ids)
        x = self.drop(x + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        return x

def test():
    net = GPT(n_embd=768, n_layer=8, n_head=8)
    x = torch.randn([1, 128, 768])
    y = net(x)

    print()

# test()