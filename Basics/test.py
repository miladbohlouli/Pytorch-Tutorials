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
from sklearn.preprocessing import StandardScaler

train_sw = SummaryWriter("logs/train/")
eval_sw = SummaryWriter("logs/eval")

class insurance(Dataset):
    def __init__(self, dataset):
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
        layers = []
        hidden_layers.insert(0, input_size)
        hidden_layers.append(1)
        for l0, l1 in zip(hidden_layers[:-1], hidden_layers[1:]):
            layers.append(nn.Linear(l0, l1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        return out

def prepare_dataset():
    dataset = pd.read_csv("../datasets/insurance.csv")
    dataset = dataset.select_dtypes(exclude=["object"]).to_numpy()
    np.random.shuffle(dataset)
    pivot = int(len(dataset) // (5/4))
    return dataset[:pivot], dataset[pivot:]


def save(dictionary, path="./save"):
    torch.save(dictionary, f"{path}/{time()}")


def load(path="./"):
    files = sorted(listdir(path))
    return torch.load(f"{path}/{files[-1]}")


if __name__ == '__main__':
    train_dataset, eval_dataset = prepare_dataset()
    train_dataset = insurance(train_dataset)
    eval_dataset = insurance(eval_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=64)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=512)

    my_model = FFNN(input_size=3, hidden_layers=[1000, 1000])
    criterian = nn.MSELoss()
    optim = torch.optim.Adam(my_model.parameters())
    epochs = 20

    for epoch in range(epochs):
        my_model.train()
        losses = []
        for x_train, y_train in train_loader:

            # forward path
            out = my_model(x_train.float())
            loss = criterian(torch.squeeze(out), y_train.float())
            losses.append(loss.item())

            # Backward path
            optim.zero_grad()
            loss.backward()
            optim.step()

        my_model.eval()
        evals = []
        for x_test, y_test in eval_loader:
            eval_out = my_model(x_test.float())
            evals.append(criterian(torch.squeeze(eval_out), y_test.float()).item())

        print(f"({epoch+1:3}/{epochs}) train: {np.mean(losses):.2f}, \teval: {np.mean(evals):.2f}")
        train_sw.add_scalar("loss", np.mean(losses), epoch)
        eval_sw.add_scalar("loss", np.mean(evals), epoch)








