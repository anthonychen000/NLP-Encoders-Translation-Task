import torch.nn as nn
import torch
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers, dropout):
        super.__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedding = self.embedding(src)
        outputs, hidden, cell = self.rnn(embedding)
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers, dropout):
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
    
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        output, (hidden, cell) = self.rnn (input, hidden, cell)
        prediction = self.fc_out(output)
        return prediction, (hidden, cell)
        
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super.__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio):
        batch_size = trg.shape[1]
        seq_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(seq_len, batch_size, trg_vocab_size)
        hidden, cell = self.encoder(src)
        input = trg[0, :]
        
        for t in range(1, seq_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs