import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import transforms
import torch.utils.tensorboard
from torch.utils import data
import os
sns.set(color_codes=True)

batch_size = 128
num_epochs = 100
learning_rate = 0.001
load = True

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
        self.Relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])

    def forward(self, x):
        out = self.fc1(x)
        out = self.Relu1(out)
        out = self.fc2(out)
        return out

def save_model(path, stuff):
    torch.save(stuff, path + "/checkpoint.pt")


def load_model(path):
    return torch.load(path + "/checkpoint.pt")


if __name__ == '__main__':
    writer = SummaryWriter("save/")
    mymodel = FFNN(28*28, [1000, 1000, 1000], 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)

    start_epoch = 0

    # loading the model if present
    if load == True:
        save_dict = load_model("save")
        optimizer.load_state_dict(save_dict["optimizer_state_dict"])
        mymodel.load_state_dict(save_dict["model_state_dict"])
        start_epoch = save_dict["epoch"] + 1
        loss = save_dict["loss"]

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    summary(mymodel, images.view(-1, 28*28).size())
    img_grid = torchvision.utils.make_grid(images[0][0])
    plt.imshow(images[0][0])
    writer.add_image("Test image", img_grid)

    for epoch in range(start_epoch, num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            # Forward pass
            images = images.reshape((-1, 28*28))
            predictions = mymodel(images)
            loss = criterion(predictions, labels)
            writer.add_graph(mymodel, images)
            writer.add_scalar("training loss", loss.item(), epoch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("The loss on epoch %d/%d: %.2f"%(epoch, num_epochs, loss))

        # Saving the model
        save_model("save", {
            "model_state_dict": mymodel.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss
        })

