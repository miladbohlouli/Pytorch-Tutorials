import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
from matplotlib import pyplot as plt
sns.set(color_codes=True)

logger = tensorboard.SummaryWriter("./logs")
dataset = sns.load_dataset("flights")


class flight_dataset(Dataset):
    def __init__(self, plot=True):
        dataset = sns.load_dataset("flights").to_numpy()

        if plot:
            figure = plt.figure(figsize=[10, 10])
            plt.plot(range(len(dataset)), dataset[:, -1])
            plt.ylabel("Passengers"), plt.xlabel("Months")
            logger.add_figure("distribution of the dataset", figure, 0)

    def __getitem__(self, item):
        pass


if __name__ == '__main__':
    data = flight_dataset()



