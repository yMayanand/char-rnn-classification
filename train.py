
import time
import random
import math
import argparse
from tqdm.notebook import tqdm


import torch
from torch import nn, optim

from model import get_model
from data import *

train_dl, val_dl = get_dl()

model = get_model(n_letters, 64, 128, n_categories)

parser = argparse.ArgumentParser('arguments for training')

parser.add_argument('--epoch', default=10, help='number of epochs to train', type=int)
args = parser.parse_args()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def validate(model):
    tot = 0
    corrects = 0
    model.eval()
    for data, labels, seq_lens in val_dl:
        bs = data.shape[0]
        out = model(data, seq_lens)
        _, idx = torch.max(out, dim=1)
        tot += bs
        corrects += torch.sum(idx==labels)
    return (corrects.item()/tot)*100
           
                        
for i in tqdm(range(args.epoch)):
    for data, labels, seq_lens in train_dl:
        optimizer.zero_grad()
        out = model(data, seq_lens)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
    print(f'epoch: {i} models:- train_loss: {loss.item()} val_acc: {validate(model)}')
    
torch.save(model.state_dict(), 'model.pt')
