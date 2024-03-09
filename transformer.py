import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length = 100):
        super().__init__()
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(
            hid_dim,
            n_heads,
            pf_dim,
            dropout
        )
        for _ in range(n_layers)])
    
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor(hid_dim))
        
    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1)
        src = self.dropout(self.tok_embedding(src) * self.scale + self.pos_embedding(pos))
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        return src
   
class EncoderLayer(nn.Module):
    def __init__():
        super().__init__()
