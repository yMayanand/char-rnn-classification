import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, vocab_len, emb_dim, 
                 hidden_size, output_size, dropout=0.):
        super(RNN, self).__init__()
        
        self.emb_size = emb_dim
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_len + 1, emb_dim, padding_idx=vocab_len)
        self.lstm = nn.LSTM(emb_dim, hidden_size, 1, batch_first=True)
        #self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, inp, seq_lens):
        bs = inp.shape[0]
        hidden = torch.zeros(1, bs, self.hidden_size)
        inp = self.emb(inp)
        data = pack_padded_sequence(inp, seq_lens, batch_first=True, enforce_sorted=False)
        out_packed, (h, c) = self.lstm(data, (hidden, hidden))
        out_padded, lengths = pad_packed_sequence(out_packed, batch_first=True)
        out = torch.flatten(torch.permute(h, (1, 0, 2)), start_dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out, out_padded


    def infer(self, inp):
        bs = inp.shape[0]
        hidden = torch.zeros(1, bs, self.hidden_size)
        inp = self.emb(inp)
        acts, (h, c) = self.lstm(inp, (hidden, hidden))
        out = torch.flatten(torch.permute(h, (1, 0, 2)), start_dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out, acts
        
def get_model(vocab_len, emb_dim, n_hidden, n_categories, dropout):
    rnn = RNN(vocab_len, emb_dim, n_hidden, n_categories, dropout=dropout)
    return rnn
