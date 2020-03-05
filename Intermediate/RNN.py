import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils import data
import time
import numpy as np

batch_size = 128
learning_rate = 0.001
sequence_length = 28
input_size = 28
num_layers = 2
num_classes = 10
epoch_num = 1
load = True
train = True

training_dataset = torchvision.datasets.MNIST(root="../datasets",
                                              train=True,
                                              transform=transforms.ToTensor())

test_dataset = torchvision.datasets.MNIST(root="../datasets",
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = data.DataLoader(training_dataset,
                               batch_size=batch_size,
                               shuffle=True)

test_loader = data.DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False)


def save_model(path, dict):
    torch.save(dict, path + "RNN_checkpoint.pt")


def load_model(path):
    return torch.load(path + "RNN_checkpoint.pt")


class RNN_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size))
        c0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size))

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
    model = RNN_model(input_size, sequence_length, num_layers, num_classes)
    criterian = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0

    if load:
        save_dict = load_model("save/")
        start_epoch = save_dict["epoch"] + 1
        model.load_state_dict(save_dict["model"])
        optimizer.load_state_dict(save_dict["optim"])
        loss = save_dict["loss"]

    if train:
        print("------------------Training the model------------------")
        for epoch in range(start_epoch, start_epoch + epoch_num):
            for i, (images, labels) in enumerate(train_loader):
                # Forward path
                predictions = model(images.view(-1, sequence_length, input_size))
                loss = criterian(predictions, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            save_model("save/", {
                "epoch":epoch,
                "model":model.state_dict(),
                "optim":optimizer.state_dict(),
                "loss":loss
            })

            print("\nEpoch %d/%d---------loss:%.2f"%(epoch+1, start_epoch + epoch_num, loss))

    # test the model
    model.eval()
    print("------------------Testing the model------------------")
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            predictions = model(images.view(-1, sequence_length, input_size)).numpy()
            predictions = np.argmax(predictions, axis=1)
            correct += np.sum(predictions == labels.numpy())
            total += labels.size(0)
        print("The test accuracy: %.2f"%(correct/total))
