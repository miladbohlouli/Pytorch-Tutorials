import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
import os
sns.set(color_codes=True)

batch_size = 128
num_epochs = 100

# load the dataset
training_dataset = torchvision.datasets.MNIST(root="../datasets/",
                                              train=True,
                                              transform=transforms.ToTensor(),
                                              download=True)

test_dataset = torchvision.datasets.MNIST(root="../datasets/",
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)

train_loader = data.DataLoader(dataset=training_dataset,
                               batch_size=batch_size,
                               shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)


class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layers[0])
        self.Relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.fc4 = nn.Linear(hidden_layers[2], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.Relu(out)
        out = self.fc2(out)
        out = self.Relu(out)
        out = self.fc3(out)
        out = self.Relu(out)
        out = self.fc4(out)
        return out


if __name__ == '__main__':
    writer = SummaryWriter("save/")
    mymodel = FFNN(28*28, [1000, 1000, 1000], 10)
    criterian = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mymodel.parameters())

    for epoch in range(num_epochs):
        for (images, labels) in train_loader:
            images = images.reshape((-1, 28*28))
            predictions = mymodel(images)
            print(predictions)
            loss = criterian(labels, predictions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("The loss on epoch %d/%d"%(epoch, num_epochs))
