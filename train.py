
import time
import random
import math
import argparse
from tqdm import tqdm


import torch
from torch import nn, optim

from .model import get_model
from .data import *

train_dl, val_dl = get_dl()

model = get_model(n_letters, n_categories)

parser = argparse.ArgumentParser('arguments for training')

parser.add_argument('--epoch', default=10, help='number of epochs to train')
args = parser.parse_args()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
