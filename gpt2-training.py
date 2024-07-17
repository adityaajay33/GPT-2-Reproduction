from dataclass import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as f

'''
Attention - A weighted sum is calculated after assigning weights to different parts of the input 
sequence based on how similar the query (Q) is to the key (K). However, it cannot catch different
relationships in the singular subspace

Multiheaded attention - It does the same thing but it captures multiple relationships by using 
several attention heads. After each head performs the attention operation the results are 
concatenated and linearly transformed

'''

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer

        '''Tokens lined up (1020 long); each token produces three vectors - query, key, and value. The 
        query and key multiply together to tell how interesting they find each other.'''
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__() #Allows it to inherit from the nn.Module super class (Polymorphism)
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)

        #Using the tanh approximation since that's what was used in the original GPT-2
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        #Two linear projections sandwich a GELU function
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(Self, x):
        x =  x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256 #max sequence length
    vocab_size: int = 65 #number of tokens: 50000 bpe merges + 256 bytes token  + 1 <[endoftext] token
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModelDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),

        ))
        self.ln_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

