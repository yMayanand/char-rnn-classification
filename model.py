import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, vocab_len, emb_dim, 
                 hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.emb_size = emb_dim
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_len + 1, emb_dim, padding_idx=vocab_len)
        self.lstm = nn.LSTM(emb_dim, hidden_size, 1, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        #self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, inp, seq_lens):
        bs = inp.shape[0]
        hidden = torch.zeros(1, bs, self.hidden_size)
        inp = self.emb(inp)
        data = pack_padded_sequence(inp, seq_lens, batch_first=True, enforce_sorted=False)
        out_packed, (h, c) = self.lstm(data, (hidden, hidden))
        #out_padded, lengths = pad_packed_sequence(h, batch_first=True)
        out = torch.flatten(torch.permute(h, (1, 0, 2)), start_dim=1)
        out = self.bn(out)
        out = self.fc(out)
        return out


    def infer(self, inp):
        bs = inp.shape[0]
        hidden = torch.zeros(1, bs, self.hidden_size)
        inp = self.emb(inp)
        out, (h, c) = self.lstm(inp, (hidden, hidden))
        out = torch.flatten(torch.permute(h, (1, 0, 2)), start_dim=1)
        out = self.bn(out)
        out = self.fc(out)
        return out
        
def get_model(vocab_len, emb_dim, n_hidden, n_categories):
    rnn = RNN(vocab_len, emb_dim, n_hidden, n_categories)
    return rnn
