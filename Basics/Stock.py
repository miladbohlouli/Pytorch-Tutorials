import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch import nn
import seaborn as sns
from matplotlib import pyplot as plt

class flight_dataset(Dataset):
    def __init__(self, window_size=12, test_count=12):
        temp = sns.load_dataset("flights")["passengers"].values
        self.test_labels = temp[-test_count:]

        mean = temp[:-test_count].mean()
        std = temp[:-test_count].std()
        self.normalization = (mean, std)

        temp = (temp - mean) / std

        self.dataset = [temp[i:i+window_size] for i in range(len(temp) - (test_count + window_size))]
        self.labels = torch.from_numpy(np.array([item for item in temp[window_size:-test_count]]))
        self.dataset = torch.from_numpy(np.stack(self.dataset, axis=0)).unsqueeze(-1)

        self.test = temp[-(test_count + window_size):-test_count]



    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

    def __len__(self):
        return len(self.dataset)

    def get_test(self):
        return self.test, self.test_labels, self.normalization


class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def initaite_hidden(self, batch_size):
        return  (torch.randn(1, batch_size, self.hidden_size),
                 torch.randn(1, batch_size, self.hidden_size))

    def forward(self, data):
        out, _ = self.lstm(data.float(), self.initaite_hidden(data.shape[0]))
        return self.linear(out[:, -1, :])

def save_model(saving_dict, saving_dir="./save"):
    torch.save(saving_dict, saving_dir + "/flight_model.pt")

def load_model(saving_dir="./save"):
    return torch.load(saving_dir+"/flight_model.pt")



# parameters definition
batch_size = 4
hidden_size = 128
window_size = 12
iterations = 10
learning_rate = 0.001
load = True

if __name__ == '__main__':
    fd = flight_dataset(window_size)

    train_loader = DataLoader(fd, batch_size=batch_size)

    my_model = Model(hidden_size)

    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    start_epoch = 0

    if load:
        print("loading the model")
        saving_dict = load_model()
        my_model.load_state_dict(saving_dict["model"])
        optimizer.load_state_dict(saving_dict["optimizer"])
        start_epoch = saving_dict["iteration"] + 1
        loss = saving_dict["loss"]

    for i in range(start_epoch, iterations + start_epoch):
        for [x_data, y_data] in train_loader:

            # Get the outputs
            outputs = my_model(x_data)

            # Calculate the loss of outputs
            loss = mse_loss(outputs.squeeze(-1), y_data.float())

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"iter:{i}/{start_epoch + iterations}\tloss:{loss:.2f}")

        saving_dict = {
            "model" : my_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": i,
            "loss": loss
        }

        save_model(saving_dict)

    my_model.eval()
    test_data, test_labels, normalization = fd.get_test()
    prediction = list(test_data)
    prediction_length = 12

    with torch.no_grad():
        for _ in range(prediction_length):
            out = my_model(torch.from_numpy(np.array(prediction[-window_size:])[None, :, None]))
            prediction.append(out.item())

    actual_predictions = (np.array(prediction[-prediction_length:]) * normalization[1]) + normalization[0]

    plt.plot(list(range(prediction_length)), actual_predictions)
    plt.plot(list(range(prediction_length)), test_labels)
    plt.legend(["predicted", "truth"]), plt.grid(alpha=0.2)
    plt.show()


