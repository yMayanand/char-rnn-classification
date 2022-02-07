import os
import glob
import math
import torch
import urllib
import zipfile
import string
import unicodedata
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

URL = 'https://download.pytorch.org/tutorial/data.zip'

urllib.request.urlretrieve(URL, 'data.zip')

with zipfile.ZipFile('data.zip', 'r') as zip_ref:
    zip_ref.extractall()

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# make dataframe for our dataset
surnames = []
categories = []
for key, value in category_lines.items():
    surnames += value
    categories += [key]*len(value)
df = pd.DataFrame({'surnames': surnames, 'categories': categories})

# all_letters, all_categories
def form_labels(df):
    label = []
    for i, j in enumerate(df['categories']):
        label.append(all_cat_dict[j])
    df['labels'] = label

all_cat_dict = {key: value for value, key  in enumerate(all_categories)}

form_labels(df)

df = df.sample(frac=1).reset_index(drop=True)

train_df = df.iloc[:math.ceil(len(df)*0.8),:].sample(frac=1).reset_index(drop=True)
val_df = df.iloc[math.ceil(len(df)*0.8):,:].reset_index(drop=True)

class Dataset:
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = list(self.df['surnames'][idx])
        for i, j in enumerate(seq):
            seq[i] = all_letters.find(j)
        label = self.df['labels'][idx]
        l = len(seq)
        return torch.tensor(seq), torch.tensor(label), l


def cust_collate(batch):
    # batch looks like [(x0,y0), (x4,y4), (x2,y2)... ]
    xs = [sample[0] for sample in batch]
    ys = [sample[1] for sample in batch] 
    zs = [sample[2] for sample in batch]
    # If you want to be a little fancy, you can do the above in one line 
    # xs, ys = zip(*samples) 
    xs = pad_sequence(xs, batch_first=True, padding_value=n_letters)
    return xs, torch.stack(ys), zs


def get_dl():
    train_ds = Dataset(train_df)
    val_ds = Dataset(val_df)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, collate_fn=cust_collate)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, collate_fn=cust_collate)

    return train_dl, val_dl
