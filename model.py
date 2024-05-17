import torch 
import torch.nn as nn

class InputEmbedding(nn.Module):
  
  def __init__(self, d_model: int, vocab_size: int):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size # how many words in the vocabulary
    self.embedding = nn.Embedding(vocab_size, d_model)
      
  def forward(self, x):
    return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
  
  
class PositionalEncoding(nn.Module):
  
  def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)
    
    # create a matrix of shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)
    # create a vector of shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
    # apply the sin to even indices in the vector
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    pe = pe.unsqueeze(0) # add a batch dimension (1, seq_len, d_model)
    
    self.register_buffer('pe', pe) # register the positional encoding buffer, so that it will be saved along with the model
  
  def forward(self, x):
    x = x + self.pe[:, :x.size(1)].requires_grad_(False) # no gradient tracking
    return self.dropout(x)
  
    