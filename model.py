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
  
  
# class LayerNorm(nn.Module):
#   # gpt copilet verion
  
#   def __init__(self, d_model: int, eps: float = 1e-6):
#     super().__init__()
#     self.d_model = d_model
#     self.eps = eps
#     self.a_2 = nn.Parameter(torch.ones(d_model))
#     self.b_2 = nn.Parameter(torch.zeros(d_model))
    
#   def forward(self, x):
#     mean = x.mean(-1, keepdim=True)
#     std = x.std(-1, keepdim=True)
#     return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
  
class LayerNomalization(nn.Module): 
  
  def __init__(self, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.tensor(1.0)) # nn.Parameter() is a learnable parameter
    self.bias = nn.Parameter(torch.tensor(1.0))
    
  def forward(self, x):
    mean = x.mean(dim = -1, keepdim = True)
    std = x.std(dim = -1, keepdim = True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias
  
  
class FeedForwardBlock(nn.Module): 
  
  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = nn.Dropout(dropout)
    self.learn1 = nn.Linear(d_model, d_ff) # W1 and B1
    self.learn2 = nn.Linear(d_ff, d_model) # We and B2
    
  def forward(self, x):
    # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model
    return self.learn2(self.dropout(torch.relu(self.learn1(x))))
  

class MultiHeadAttentionBlock(nn.Module): 
  
  def __init__(self, d_model: int, head: int, dropout: float) -> None: 
    super().__init__()
    self.d_model = d_model
    self.head = head
    assert d_model % head == 0, "d_model must be divisible by h"
    
    self.d_k = d_model // head
    
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)
    
    self.dropout = nn.Dropout(dropout)
    
  @staticmethod # static method, no need to create an instance of the class to call the function 
  def attention(query, key, value, mask, dropout: nn.Dropout): 
    d_k = query.shape[-1]
    
     # (batch, head, seq_len, d_k) --> (batch, head, seq_len, seq_len)
     # @ here is the matrix multiplication
    attention_scores = (query @ key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None: 
      attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
    attention_scores = torch.softmax(attention_scores, dim = -1) # (batch, head, seq_len, seq_len)
    if dropout is not None: 
      attention_scores = dropout(attention_scores)
    
    return (attention_scores @ value, attention_scores) # (batch, head, seq_len, d_k), (batch, head, seq_len, seq_len)
    
  
  def forward(self, q, k, v, mask): 
    # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
    query = self.w_q(q) 
    key = self.w_k(k) 
    value = self.w_v(v) 
    
    # (batch, seq_len, d_model) --> (batch, seq_len, head, d_k) --> (batch, head, seq_len, d_k)
    # by doing transpose, we can need each head to see all the sequence length
    query = query.view(query.shape[0], query.shape[1], self.head, self.d_k).transpose(1,2)
    key =  key.view(key.shape[0], key.shape[1], self.head, self.d_k).transpose(1,2)
    value = value.view(value.shape[0], value.shape[1], self.head, self.d_k).transpose(1,2)
    
    x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
    
    # (batch, head, seq_len, d_k) --> (batch, seq_len, head, d_k) --> (batch, seq_len, d_model)
    # Transposing a tensor can make its underlying memory layout non-contiguous. That is, the elements in the tensor might not be stored in a single continuous block of memory.
    # contiguous() returns a contiguous tensor containing the same data. If the tensor is already contiguous, it does nothing. If not, it creates a copy of the tensor that is stored contiguously in memory.
    x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.head * self.d_k)
    
    # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
    return self.w_o(x)


class ResidualConnection(nn.Module): 
  
  def __init__(self, dropout: float) -> None: 
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNomalization()
    
  def forward(self, x, sublayer): 
    '''
    sublayer is a callable function for showing flexibility and encapsulation
    Encapsulation: By encapsulating the normalization and dropout within the ResidualConnection module, any changes to the sequence 
      of these operations can be easily managed within the module without modifying the calling code.
    Maintainability: If you decide to change how normalization or dropout is applied, or if additional steps are needed, you can do 
      so within the ResidualConnection class without altering the encoder block logic.
    ''' 
    return x + self.dropout(sublayer(self.norm(x))) # doing normalization before the sublayer in most implementations, which are different from the original paper
  

class EncoderBlock(nn.Module): 
  
  def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None: 
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
  def forward(self, x, src_mask): 
    # src_mask is used to mask the padding tokens from inputs
    
    '''
    Why not x = self.residual_connections[0](x, self.self_attention_block(x, x, x, src_mask))
    because self.self_attention_block(x, x, x, src_mask) directly compute the outputs, 
    but the residual connection needs iputs from sublayers, which should be nn.Module layers, rather than tensor
    
    The lambda function is used to delay the execution of the self-attention block until it is called within the 
      ResidualConnection module, ensuring the proper application of normalization and dropout.
    '''
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
    x = self.residual_connections[1](x, self.feed_forward_block)
    return x 

class Encoder(nn.Module): 
  
  def __init__(self, layers: nn.Modulelist) -> None: 
    super().__init__()
    self.layers = layers
    self.norm = LayerNomalization()
    
  def forward(self, x, src_mask): 
    for layer in self.layers: 
      x = layer(x, src_mask)
    return self.norm(x)   


class DecoderBlock(nn.Module): 
  
  def __init__(self, self_attention_block: MultiHeadAttentionBlock, corss_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None: 
    super().__init__()
    self.self_attention_block = self_attention_block
    self.corss_attention_block = corss_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
  def forward(self, x, encoder_output, src_mask, tgt_mask): 
    # src_mask/encoder_mask is used to mask the padding tokens from inputs
    # tgt_mask/decoder_mask is used to mask the future tokens from the decoder output
    
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
    x = self.residual_connections[1](x, lambda x: self, self.corss_attention_block(x, encoder_output, encoder_output, src_mask))
    x = self.residual_connections[2](x, self.feed_forward_block)
    return x
  
class Decoder(nn.Module): 
  
  def __init__(self, layers: nn.ModuleList) -> None: 
    super().__init__()
    self.layers = layers
    self.norm = LayerNomalization()
    
  def forward(self, x, encoder_output, src_mask, tgt_mask): 
    for layer in self.layers: 
      x = layer(x, encoder_output, src_mask, tgt_mask)
    return self.norm(x)
  

class ProjectionLayer(nn.Module): 
  
  def __init__(self, d_model: int, vocab_size: int) -> None: 
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.proj = nn.Linear(d_model, vocab_size)
    
  def forward(self, x): 
    # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
    # log_softmax is used to enhance numerical stability when handling small values in the softmax function, which used to lead to underflow
    return torch.log_softmax(self.proj(x), dim = -1)


class Transformer(nn.Module): 
  
  def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None: 
    # scr_embed and tgt_embed are used to convert between different languages 
    
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.src_pos = src_pos
    self.tgt_pos = tgt_pos
    self.projection_layer = projection_layer
    
  def encode(self, src, src_mask): 
    return self.encoder(self.src_embed(src), src_mask)
  
  def decode(self, tgt, encoder_output, src_mask, tgt_mask): 
    return self.decoder(self.tgt_pos(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask)
    
  def project(self, x): 
    return self.projection_layer(x)
  

# def build_transformer()
  
  
  