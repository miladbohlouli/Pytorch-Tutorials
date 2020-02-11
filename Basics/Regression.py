import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.set(color_codes=True)


def build_dataset(function, noise_intensity=1, range=[-10, 10], dataset_size = 10000):
    x = np.random.uniform(range[0], range[1], dataset_size).astype(np.float32)
    y = function(x)
    noise = np.random.normal(0, noise_intensity, dataset_size).astype(np.float32)
    noisy_y = np.copy(y + noise)
    return x, noisy_y, y


def plot(x, y, color="purple"):
    plt.scatter(x, y, s=2, color=color)
    plt.grid(alpha=0.7)
    plt.xlabel("Regressor"), plt.ylabel("Target feature")


class Linear_regression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Linear_regression, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.hidden(x)
        out = self.output(out)
        return out

class regression(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, out):
        super(regression, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.output = nn.Linear(hidden2, out)

    def forward(self, x):
        out = self.hidden1(x)
        out = self.hidden2(out)
        return self.output(out)

num_epochs = 500
learning_rate = 0.01
training_size = 1000

if __name__ == '__main__':
    sin = lambda x: np.sin(x)
    linear = lambda x: 10 * x  + 2
    x, y, _ = build_dataset(linear, noise_intensity=5, dataset_size=training_size)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    inputs, targets = torch.from_numpy(x_train), torch.from_numpy(y_train)
    inputs, targets = inputs.view((-1, 1)), targets.view((-1, 1))
    # mymodel = Linear_regression(1, 1000, 1)
    mymodel = regression(1, 1000, 1000, 1)
    criterian = nn.MSELoss()
    optimizer = torch.optim.Adam(mymodel.parameters())

    for epoch in range(num_epochs):
        predictions = mymodel(inputs)
        loss = criterian(predictions, targets)

        # Optimizing the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Operating epoch %d out of %d. loss %.2f"%(epoch, num_epochs, loss))

# 09143921870

    inputs_test = torch.from_numpy(x_test).view((-1, 1))
    targets_test = torch.from_numpy(y_test).view((-1, 1))
    predictions = mymodel(inputs_test).detach().numpy()
    plot(x_train, y_train)
    plot(x_test, predictions, color="yellow")
    plt.show()