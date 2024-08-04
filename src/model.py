import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    
    def __init__(self, d_model, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size  
        self.embedding = nn.Embedding(vocab_size, d_model)
        # pytorch provides Embedding layer
        
    def forward(self, x):
        # map the embedding with the pytorch Emb
        return self.embedding(x) * math.sqrt(self.d_model) 
    
    
# lets now build the positional embeddings
# embeddign size 512, vocab size - final positional matrix
    
class PositionalEmbedding(nn.Module):
    
    def __init__(self, d_model:int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # seqence lenght - dimension of the model matrix using these 2
        pe = torch.zeros(seq_len, d_model)
        # creating vector of shape seq_lenth
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

