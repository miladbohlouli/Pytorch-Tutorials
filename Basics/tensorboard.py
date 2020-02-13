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

train_data = torchvision.datasets.MNIST("../datasets/",
                                        train=True,
                                        download=False)

test_data = torchvision.datasets.MNIST("../datasets/",
                                        train=False,
                                        download=False)


writer = SummaryWriter("save/")

for i in range(10):
    pass