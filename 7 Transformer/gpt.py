# imports
import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
block_size = 256 # maximum context length for prediction
batch_size = 64 # number of independant sequences we will process in paralell
max_iters = 5000 # maximum number of training steps
learning_rate = 3e-4 # learning rate for the optimizer
eval_interval = 500 # number of training steps between evaluations
eval_iters = 200 # number of minibatches we use to estimate the loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
num_heads = 6
num_blocks = 6
dropout_rate = 0.2
print(f"Running on {device}\n")


# random seed for reproducability
torch.manual_seed(5)


# read data
with open('7 Transformer/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
  text = f.read()


# tokenizer (encoder/decoder)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# train\test splits
data = torch.tensor(encode(text), dtype=torch.int64)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# data loading (minibatches)
def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data)-block_size, size=(batch_size,))
  x = torch.stack([data[i : i+block_size] for i in ix])
  y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y


# single attention head
class Head(nn.Module):
  
  def __init__(self, input_size:int, head_size:int) -> None:
    super().__init__()
    self.key = nn.Linear(input_size, head_size, bias=False)
    self.query = nn.Linear(input_size, head_size, bias=False)
    self.value = nn.Linear(input_size, head_size, bias=False)
    # since tril is not a parameter, we assign it to the module using the registry buffer, as per PyTorch
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout_rate)
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    # compute attention-scores ("affinities")
    k = self.key(x) # (B, T, C) @ (C, head_size) ---> (B, T, head_size)
    q = self.query(x) # (B, T, C) @ (C, head_size) ---> (B, T, head_size)
    affinities = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) ---> (B, T, T)
    # compute weights
    weights = affinities.masked_fill(self.tril[:T, :T]==0, float('-inf')) # type: ignore
    # TODO why did accidentally making the softmax dim=1 reduce the loss dramatically, is it look because of look ahead?
    # TODO: Understand the relationship between each individual weight and the output
    weights = F.softmax(weights, dim=-1) # (B, T, T)
    weights = self.dropout(weights) # randomly prevent some of the nodes (tokens) from communicating
    # perform weighted aggregation of the vlaues
    v = self.value(x) # (B, T, C) @ (C, head_size) ---> (B, T, head_size)
    out = weights @ v # (B, T, T) @ (B, T, head_size) ---> (B, T, head_size)
    return out


# multihead attention
class MultiHeadAttention(nn.Module):
  
  def __init__(self, input_size:int, num_heads:int, head_size:int) -> None:
    super().__init__()
    self.heads = nn.ModuleList([Head(input_size, head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(num_heads*head_size, num_heads*head_size)
    self.dropout = nn.Dropout(dropout_rate)
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    # x has shape (B, T, C)
    out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, num_heads*head_size)
    out = self.proj(out) # linear projection
    out = self.dropout(out) # randomly dropout the output
    return out


# feed forward
class FeedForward(nn.Module):
  
  def __init__(self, input_size:int, ffwd_hidden_size:int, output_size:int) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_size, ffwd_hidden_size),
      nn.ReLU(),
      nn.Linear(ffwd_hidden_size, output_size), # linear projection
      nn.Dropout(dropout_rate), # randomly dropout the output
    )
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    # x has shape (B, T, input_size)
    out = self.net(x) # (B, T, output_size)
    return out


# transformer
class TransformerBlock(nn.Module):
  
  def __init__(self, input_size:int, num_heads:int, head_size:int, ffwd_hidden_size:int, output_size:int) -> None:
    super().__init__()
    self.ln1 = nn.LayerNorm(input_size) # pre-norm formulation (deviation from the original transformer paper)
    self.sa = MultiHeadAttention(input_size, num_heads, head_size)
    self.ln2 = nn.LayerNorm(num_heads*head_size)
    self.ffwd = FeedForward(num_heads*head_size, ffwd_hidden_size, output_size)
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    # x has shape (B, T, C)
    x = self.ln1(x) # (B, T, C)
    x = x + self.sa(x) # (B, T, num_heads*head_size) # the "+" is for the skip connection
    x = self.ln2(x) # (B, T, num_heads*head_size)
    x = x + self.ffwd(x) # (B, T, output_size)
    return x


# bigram language model
class LanguageModel(nn.Module):
  
  def __init__(self) -> None:
    super().__init__()
    # the embedding for each token is simply the logits for the next token
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(
      *[
        TransformerBlock(
          input_size=n_embd, 
          num_heads=num_heads, 
          head_size=n_embd//num_heads, 
          ffwd_hidden_size=n_embd*4, 
          output_size=n_embd
        ) for _ in range(num_blocks)
      ]
    )
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx:torch.Tensor, targets:torch.Tensor|None = None) -> tuple[torch.Tensor, torch.Tensor|None]:
    B, T = idx.shape # idx and targets are both (B,T) tensors of integers
    tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
    x = tok_emb + pos_emb # (B, T, n_embd) + (T, n_embd) ---> (B, T, n_embd)
    x = self.blocks(x) # (B, T, n_embd)
    logits = self.lm_head(x) # (B, T, vocab_size)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      # merge the batch and time dimension since F.cross_entropy expects only one "batch" dimension
      loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    
    return logits, loss
  
  def generate(self, idx:torch.Tensor, max_new_tokens:int) -> torch.Tensor:
    # idx is (B,T) tensor of integers of current context
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:] # crop the context to a maximum of block_size
      logits, _loss = self(idx_cond)
      # we only care about the predicted token for the last time-step
      logits = logits[:, -1, :] # (B, C)
      probs = F.softmax(logits, dim=1)
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx


# estimate loss
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval() # switch the model into eval mode
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
      Xb, Yb = get_batch(split)
      logits, loss = model(Xb, Yb)
      losses[i] = loss.item()
    out[split] = losses.mean()
  model.train() # switch the model back into train mode
  return out


# initialize model and optimizer
model = LanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # you can get away with larger lr for smaller networks


# training loop
for iter in range(max_iters):
  
  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


# generate from the model
context_idx = torch.zeros((1, 1), dtype=torch.int64, device=device)
print(decode(model.generate(context_idx, max_new_tokens=500)[0].tolist()))