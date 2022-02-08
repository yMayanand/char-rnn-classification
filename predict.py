import torch
import argparse
from model import get_model
from data import *

parser = argparse.ArgumentParser()

parser.add_argument('--emb_size', default=32, type=int, help='embedding size')
parser.add_argument('--hidden_size', default=64, type=int, help='hidden size')
parser.add_argument('--ar', default=0, type=float, help='activity regularisation constant')
parser.add_argument('--dropout', default=0, type=float, help='dropout value')
args = parser.parse_args()



model = get_model(n_letters, args.emb_size, args.hidden_size, n_categories, dropout=args.dropout)
model.load_state_dict(torch.load('./checkpoints/model.pt'))


def predict(name):
    seq = list(name)
    for i, j in enumerate(seq):
        seq[i] = all_letters.find(j)
    seq_len = len(seq)
    inp = torch.tensor(seq).unsqueeze(0)
    model.eval()
    out = model.infer(inp)
    _, idx = torch.max(out, dim=1)
    return all_categories[idx.item()]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="hinton", help="enter the name to predict country of origin")
    args = parser.parse_args()
    print(f'prediction for {args.name} is {predict(args.name)}')

