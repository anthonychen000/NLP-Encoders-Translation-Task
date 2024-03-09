import torch.nn as nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(emb_dim + 2 * enc_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return output, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(2 * enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_output):
        batch_size = encoder_output.shape[1]
        src_len = encoder_output.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_output = encoder_output.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_output), dim = 2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim = 1)
    
class Decoder(nn.Module):
    def __init__(self, attention, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + 2 * enc_hid_dim, dec_hid_dim)
        self.fc = nn.Linear(emb_dim + 2 * enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden, encoder_output, input):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        a = self.attention(hidden, encoder_output)
        a = a.unsqueeze(1)
        encoder_output = encoder_output.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_output)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden)
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc(torch.cat((output, weighted, embedded), dim = 1))
        return prediction, hidden.squeeze(0)
    
