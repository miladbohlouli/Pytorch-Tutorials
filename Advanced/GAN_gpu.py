import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize", default=64, type=int)
parser.add_argument("--imagesize", default=28*28, type=int)
parser.add_argument("--noise_size", default=100, type=int)

# Generator parameters
parser.add_argument("--generator_structure", default=[128, 128, 128], type=list)
parser.add_argument("--generator_activation", default="Relu", type=str)
parser.add_argument("--g_steps", default=1, type=int)

# Discriminator parameters
parser.add_argument("--generator_structure", default=[128, 128, 128], type=list)
parser.add_argument("--generator_activation", default="Relu", type=str)
parser.add_argument("--d_steps", default=1, type=int)

def make_mlp(layers: list):
    nn_layers = []
    for dim_in, dim_out in zip(layers[:-1], layers[1:]):
        nn_layers.append(nn.Linear(dim_in, dim_out))
        if

def init_weights(m):
    if m.__class__.__name__.find("Linear"):
        nn.init.kaiming_normal(m.weight)


# define the model
class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        make_mlp()



def main(args):

    dataset_train = MNIST(root="../datasets/",
                          train=True,
                          transform=ToTensor())

    dataset_test = MNIST(root="../datasets/",
                         train=False,
                         transform=ToTensor())

    train_loader = DataLoader(dataset_train,
                              batch_size=args.batchsize,
                              shuffle=True)

    test_loader = DataLoader(dataset_test,
                             batch_size=args.batchsize,
                             shuffle=True)

    g = Generator(args)

    for x_train, labels in train_loader:
        pass


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
