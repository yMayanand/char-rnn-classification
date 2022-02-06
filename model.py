import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_len, emb_dim, 
                 hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.emb_size = emb_dim
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_len + 1, emb_dim, padding_idx=vocab_len + 1)
        self.lstm = nn.LSTM(emb_dim, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, seq_lens):
        bs = input.shape[0]
        hidden = torch.zeros(1, bs, self.hidden_size)
        input = self.emb(input)
        data = pack_padded_sequence(input, seq_lens, batch_first=True, enforce_sorted=False)
        out_packed, (h, c) = self.lstm(data, (hidden, hidden))
        out_padded, lengths = pad_packed_sequence(out, batch_first=True)
        store_out = []
        for i, j in enumerate(lengths):
            store.append(out_padded[i, j.item()-1])
        out = torch.stack(store_out)

        out = self.fc(out)
        return out
def get_model(vocab_len, emb_dim, n_hidden, n_categories):
    rnn = RNN(vocab_len, emb_dim, n_hidden, n_categories)
    return rnn
