# imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
block_size = 8 # maximum context length for prediction
batch_size = 32 # number of independant sequences we will process in paralell
max_iters = 3000 # maximum number of training steps
learning_rate = 1e-2 # learning rate for the optimizer
eval_interval = 300 # number of training steps between evaluations
eval_iters = 200 # number of minibatches we use to estimate the loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# bigram language model
class BigramLanguageModel(nn.Module):
  
  def __init__(self, vocab_size:int) -> None:
    super().__init__()
    # the embedding for each token is simply the logits for the next token
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx:torch.Tensor, targets:torch.Tensor|None = None) -> tuple[torch.Tensor, torch.Tensor|None]:
    # idx and targets are both (B,T) tensors of integers
    logits = self.token_embedding_table(idx) # (B, T, C)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.size()
      # merge the batch and time dimension since F.cross_entropy expects only one "batch" dimension
      loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    
    return logits, loss
  
  def generate(self, idx:torch.Tensor, max_new_tokens:int) -> torch.Tensor:
    # idx is (B,T) tensor of integers of current context
    for _ in range(max_new_tokens):
      logits, _loss = self(idx)
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
model = BigramLanguageModel(vocab_size).to(device)
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
context_idx = torch.zeros((1, 1), dtype=torch.int64).to(device)
print(decode(model.generate(context_idx, max_new_tokens=500)[0].tolist()))