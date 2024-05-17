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
  
  def __init__(self, d_model: int, max_seq_len: int, dropout: float) -> None: # return None from __init__ method, to enhance understanding and clarity
    super().__init__()
    self.d_model = d_model
    self.max_seq_len = max_seq_len
    self.dropout = nn.Dropout(dropout)
    
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1) # add 1 in the dimension index 1 to make it (max_seq_len, 1)
    '''
    Why the div_term is computed in this way? different from the original paper?
      - Numerical Stability: Direct computation of very large or very small powers can lead to numerical instability due to floating-point precision limitations.
      - Efficiency: Exponentiation with large bases and exponents can be computationally expensive.
    '''
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / d_model)) 
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    pe = pe.unsqueeze(0) # add a batch dimension (1, max_seq_len, d_model)
    
    self.register_buffer('pe', pe) # register the positional encoding buffer, so that it will be saved along with the model
  
  def forward(self, x):
    x = x + self.pe[:, :x.size(1)].requires_grad_(False) # x.size(1) is the current seq_len, which is less than max_seq_len initialized # no gradient tracking
    return self.dropout(x)
  
    