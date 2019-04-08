import os
from torchtext import data
from torchtext import datasets

folder = os.environ['DATA_DIRECTORY']

inputs = data.Field(lower=True, tokenize='spacy')
answers = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, answers, root=folder)

print('Data set is ready')
