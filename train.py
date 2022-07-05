import time
import random
import math
import argparse

import torch
from torch import nn, optim

from model import get_model
from data import *


parser = argparse.ArgumentParser('arguments for training')

parser.add_argument('--epoch', default=10, help='number of epochs to train', type=int)
parser.add_argument('--bs', default=32, type=int, help='batch size for training')
parser.add_argument('--lr', default=1e-3, type=float, help='controls learning rate of model')
parser.add_argument('--opt', default='Adam', type=str, help='optimizer for training')

parser.add_argument('--emb_size', default=512, type=int, help='embedding size')
parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')

parser.add_argument('--wd', default=0, type=float, help='weight decay parameter')
parser.add_argument('--ar', default=0, type=float, help='activity regularisation constant')
parser.add_argument('--dropout', default=0, type=float, help='dropout value')
parser.add_argument('--label_smoothing', default=0, type=float, help='label smoothing parameter for loss')


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model(n_letters, args.emb_size, args.hidden_size, n_categories, args.dropout).to(device)

weights = [4.702148925537231,
           33.84532374100719,
           2.564458980648678,
           125.45333333333333,
           46.122549019607845,
           13.252112676056338,
           93.15841584158416,
           12.977931034482758,
           9.484879032258064,
           34.97769516728624,
           127.14864864864865,
           67.20714285714286,
           31.468227424749163,
           1.0,
           40.38197424892704,
           99.0421052631579,
           18.09423076923077,
           31.573825503355703]

weights = torch.tensor(weights, device=device)

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
train_dl, val_dl = get_dl(args.bs)
optimizer = getattr(optim, args.opt, optim.Adam)
if optimizer.__name__ == 'SGD':
    optimizer = optimizer(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
else:
    optimizer = optimizer(model.parameters(), lr=args.lr, weight_decay=args.wd)

#scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dl), epochs=args.epoch)

def validate(model):
    tot = 0
    corrects = 0
    model.eval()
    for data, labels, seq_lens in val_dl:
        data = data.to(device)
        labels = labels.to(device)
        seq_lens = torch.tensor(seq_lens).to('cpu')
        
        bs = data.shape[0]
        out, acts  = model(data, seq_lens)
        _, idx = torch.max(out, dim=1)
        tot += bs
        corrects += torch.sum(idx==labels).item()
    print(f"correct/total: {corrects}/{tot}")
    
    return (corrects/tot)*100
           
                        
for i in range(args.epoch):
    model.train()
    for data, labels, seq_lens in train_dl:
        data = data.to(device)
        labels = labels.to(device)
        seq_lens = torch.tensor(seq_lens).to('cpu')
        
        
        optimizer.zero_grad()
        out = model(data, seq_lens)
        loss = criterion(out[0], labels) + args.ar*torch.mean(torch.pow(out[1], 2))
        loss.backward()
        optimizer.step()
        #scheduler.step()
    print(f'epoch: {i} models:- train_loss: {loss.item()} val_acc: {validate(model)}')
    
torch.save(model.state_dict(), './checkpoints/model.pt')
