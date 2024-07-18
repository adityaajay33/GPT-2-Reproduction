from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

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
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
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
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x =  x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx): #The token indices are being fed to the model

        #index is of shape: (Batch_size, Token_size)
        B, T = idx.size()
        #T has to be les than max sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        #arange is like range but for pytorch, and youre iterating from 0 to T to make a position index
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #last param makes sure you're training on gpu and cpu
        pos_emb = self.transformer.wpe(pos) #position embeddings of shape (T, n_embd). Identical for each row
        tok_emb = self.transformer.wte(idx) #token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb #they add up at each step

        #forward the transformer blocks
        for block in self.transformer.h:
            x = block(x)

        #the final layer norm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        '''If the input is B x T indices, then at every single B x T you will calculate the token
        for what comes next in the sequence, what is  (B, T+1). Vocab is the number of possible tokens.
        After softmax the logits become probabilities.'''
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

model = GPT.from_pretrained('gpt2')

num_return_sequences = 5
max_length = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT.from_pretrained('gpt2')
model.eval() #there might be no effect since there are no training specific layers like dropout or batch norm
model.to(device) #moving to gpu

#prefix tokens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,") #Encodes and writtens a list of integers, tokens are string chunks
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #repeats the same number of times as the number of lines you asked for output
x = tokens.to(device) #x is the index you put in the idx, to get the logits

#X is of shape (B, T), where B = batch_size, T = time
torch.manual_seed(42)
if device=="cuda":
    torch.cuda.manual_seed(42)

while x.size(1) < max_length:#each loop iteration adds one more column to x, more data comes along through sampling
    with torch.no_grad(): #saves a lot of space and time, because youre not caching a lot of data. data is normally cached if youre going to calculate gradients, which we are not
        logits = model(x) 

        #get logits at last location/column only 
        logits = logits[:, -1, :]
        #get probability using softmax from logits
        probs = F.softmax(logits, dim=-1)
        #did a topk sampling of 50; huggingface pipeline (keep only top 50 probabilities, and everything below we claim as 0 and renormalize)
        #it helps keep the model stick to the vicinity of likely results 
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        #select a token from the top 50 probabilities
        ix = torch.multinomial(topk_probs, 1)

        xcol = torch.gather(topk_indices, -1, ix)
        #append it to the sequence as a column
        x = torch.cat((x, xcol), dim=1)
        #x is of size 5 x 30


for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)