import torch
import argparse
from torch.nn.utils.rnn import pad_sequence,
from model import get_model
from data import *

parser = argparse.ArgumentParser()
parser.add_argument(('--name', default='hinton', help='enter the name to predict country of origin')
args = parser.parse_args()

model = get_model(n_letters, 64, 128, n_categories)
model.load_state_dict(torch.load('model.pt'))

seq = list(args.name)
for i, j in enumerate(seq):
    seq[i] = all_letters.find(j)
seq_len = len(seq)
inp = torch.tensor(seq)
model.eval()
out = model(inp)
_, idx = torch.max(out, dim=1)
idx = torch.topk(x, 1)[1].item()
