import numpy as np
from matplotlib import pyplot as plt
# import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch
from time import time
from os import listdir

class insurance(Dataset):
    def __init__(self):
        dataset = pd.read_csv("../datasets/insurance.csv")
        dataset = dataset.select_dtypes(exclude=["object"]).to_numpy()
        self.X, self.Y = dataset[:, :-1], dataset[:, -1]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

    def __str__(self):
        return str(self.X.shape)


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(FFNN, self).__init__()
        self.layers = []
        hidden_layers.insert(0, input_size)
        hidden_layers.append(1)
        for l0, l1 in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.layers.append(nn.Linear(l0, l1))


def save(dictionary, path="./save"):
    torch.save(dictionary, f"{path}/{time()}")


def load(path="./"):
    files = sorted(listdir(path))
    return torch.load(f"{path}/{files[-1]}")


if __name__ == '__main__':
    dataset = insurance()
    train_loader = DataLoader(dataset,
                              batch_size=64)

    my_model = FFNN(input_size=3, hidden_layers=[1000, 1000])
    crriterian = nn.MSELoss()
    optim = torch.optim.Adam()





