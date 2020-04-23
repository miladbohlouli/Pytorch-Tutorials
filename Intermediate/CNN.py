import torch
import numpy as np
import os
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils import data
import torchvision
from tqdm import *

batch_size = 64
num_epochs = 1
save_dir = "save/"
learning_rate = 0.001
load = True

train_dataset = torchvision.datasets.MNIST("../datasets",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)

test_dataset = torchvision.datasets.MNIST("../datasets",
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=False)

train_loader = data.DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True)

test_loader = data.DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=True)


def save_model(path, state_dict):
    torch.save(state_dict, path + "CNN.pt")


def load_model(path):
    return torch.load(path + "CNN.pt")


class CNN_module(nn.Module):
    def __init__(self, num_classes):
        super(CNN_module, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        return self.layer3(out)


if __name__ == '__main__':

    model = CNN_module(num_classes=10)
    criterian = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0


    if load:
        dictionary = load_model("save/")
        model.load_state_dict(dictionary["model"])
        optimizer.load_state_dict(dictionary["optimizer"])
        start_epoch = dictionary["epoch"] + 1
        loss = dictionary["loss"]

    # training the model
    for epoch in range(start_epoch, start_epoch + num_epochs):
        for (images, labels) in tqdm(train_loader):

            # get output from model
            prediction = model(images)

            # calculate the loss
            loss = criterian(prediction, labels)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save the model
        save_model(save_dir,
        {
            "epoch":epoch,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "loss":loss
        })
        print("\nEpoch %d/%d---------loss:%.4f\n" % (epoch + 1, start_epoch + num_epochs, loss))

    # Test the model
    model.eval()
    print("-------------Testing the model-------------")
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in tqdm(test_loader):
            predictions = model(images).numpy()
            predicted_labels = np.argmax(predictions, axis=1)
            correct += np.sum(predicted_labels == labels.numpy())
            total += labels.size(0)

        print("total accuracy: %.2f" % (correct / total))

