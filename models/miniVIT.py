import math
import logging
 
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, resid_pdrop, attn_pdrop,  n_head):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, resid_pdrop, attn_pdrop,  n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, resid_pdrop, attn_pdrop,  n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd*2),
            nn.GELU(),
            nn.Linear(n_embd*2, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VIT(nn.Module):
    def __init__(self, n_embd, n_layer, n_head, num_pos_feats=288//3, attn_pdrop=0.1, embd_pdrop=0.1, resid_pdrop=0.1, down_ratio=1):
        super().__init__()
        # input embedding stem
        # self.proj = nn.Liear(512, n_embd, bias=False)
        self.pos_emb = PositionEmbeddingSine(num_pos_feats=num_pos_feats) # 288
        self.drop = nn.Dropout(embd_pdrop)
        self.down = down_ratio
        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, resid_pdrop, attn_pdrop,  n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

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
        b, seq_len, c, w, h = x.size()

        x = x.view(b*seq_len, c, w, h)
        x = nn.Unfold((self.down, self.down), stride=self.down)(x) # b, c*k*k,w/3 * h/3
        x = x.view(b, seq_len, x.shape[-2], x.shape[-1])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(b, x.shape[1]*x.shape[2], x.shape[3])
        # pos
        pos_mask = torch.ones((b, seq_len, w//self.down, h//self.down)).to(x.device)
        pos_embd = self.pos_emb(pos_mask)
        pos_embd = pos_embd.view(b, -1, pos_embd.shape[-1])
        pos_embd = F.interpolate(pos_embd, x.shape[-1], mode='linear')
        #
        x = self.drop(x+pos_embd)
        x = self.ln_f(self.blocks(x))
        x = x.reshape(b, seq_len, x.shape[1]//seq_len, x.shape[2]).permute(0, 1, 3, 2)
        x = x.view(b*seq_len, x.shape[-2], x.shape[-1])
        x = nn.Fold(output_size=(w, h), kernel_size=(self.down, self.down), stride=self.down)(x)
        x = x.view(b, seq_len, c, w, h)
        return x

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=288//3, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        t_embed = mask.cumsum(1, dtype=torch.float32)
        y_embed = mask.cumsum(2, dtype=torch.float32)
        x_embed = mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            t_embed = t_embed / (t_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_t = t_embed[:, :, :, :, None] / dim_t

        pos_t = torch.stack((pos_t[:, :, :, :, 0::2].sin(), pos_t[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos = torch.cat((pos_t, pos_y, pos_x), dim=4)
        return pos

def test():
    net = VIT(n_embd=96, n_layer=3, n_head=4)
    x = torch.randn([2, 25, 96, 6, 6])  # b, seq_len, c, w, h = x.size()
    net(x)
    
    # net = PositionEmbeddingSine()
    # x = torch.randn([1, 5, 512])
    # mask = torch.ones([1, 12, 32])
    # y = net(x, mask)

    print()

# test()