import torch
import torchvision as tv
from torchvision import transforms
from torch.utils import data
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import time
import os

sw = SummaryWriter("./checkpoint/")

dataset_train = tv.datasets.MNIST("../datasets",
                                  train=True,
                                  transform=transforms.ToTensor(),
                                  download=True)


dataset_test = tv.datasets.MNIST("../datasets",
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)


train_loader = data.DataLoader(dataset_train,
                               batch_size=16,
                               shuffle=True)


test_loader = data.DataLoader(dataset_train,
                               batch_size=16,
                               shuffle=False)


def save_model(dictionary, path="."):
    torch.save(dictionary, f"{path} + {time.time()}")

def load_model(path="./save"):
    models = sorted(os.listdir(path))
    return torch.load(f"{path}/{models[-1]}")


class FFNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        layers1 = self.relu(self.fc1(x))
        return self.fc2(layers1)


if __name__ == '__main__':

    mymodel = FFNN(28*28, 10)
    optimizer = torch.optim.Adam(mymodel.parameters())
    loss = nn.CrossEntropyLoss()
    num_epochs = 15

    for epoch in range(100):
        for i, (x, y) in enumerate(train_loader):

            # Forward path
            x = x.reshape((-1, 28*28))
            outputs = mymodel(x)
            ls = loss(outputs, y)

            # Backward path
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()

        print(f"The loss in epoch {epoch}: {ls:.2}")
        sw.add_scalar("training_loss", ls)




