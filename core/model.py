import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .blocks import TransformerBlock
from .embeddings import PositionalEmbedding

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, d_ff=3072, 
                 dropout=0.1, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(d_model, max_seq_len)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (optional)
        self.head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        # Embed tokens and positions
        x = self.token_embedding(x) * (self.d_model ** 0.5)
        x = self.pos_embedding(x)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x)
            
        # Final output
        x = self.norm(x)
        return self.head(x)