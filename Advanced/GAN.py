import torch
from matplotlib import pyplot as plt
import numpy
import torch.nn as nn
from sklearn.datasets import  load_digits
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


# parameters
LOG_DIR = "./logs"
SAVE_DIR = "./save"
hidden_size = 128
batch_size = 16
num_epochs = 3900
noise_size = 100
image_size = 8 * 8
load = True

D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
)

G = nn.Sequential(
    nn.Linear(noise_size, hidden_size),
    nn.LeakyReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
)


class mnist_dataset(Dataset):
    def __init__(self, images):
        self.images = torch.from_numpy((images / 16).astype(np.float32))

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return self.images.size()[0]


summary = SummaryWriter(log_dir=LOG_DIR)

# optimizers
d_optimizer = torch.optim.Adam(D.parameters(), lr = 0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr = 0.0002)

# criterian
criterian = nn.BCELoss()

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

def save_model(save_dir, save_dict):
    torch.save(save_dict, os.path.join(save_dir, "checkpoint.pt"))

def load_model(save_dir):
    return torch.load(os.path.join(save_dir, "checkpoint.pt"))

if __name__ == '__main__':
    MNIST = load_digits()
    fake_results = None
    start_epoch = 0

    # create the train and test dataset
    data = mnist_dataset(MNIST["data"])

    # create the loaders
    data_loader = DataLoader(data, shuffle=False, batch_size=batch_size)

    if load:
        saving_dict = load_model(SAVE_DIR)
        D.load_state_dict(saving_dict["discriminator"])
        G.load_state_dict(saving_dict["generator"])
        d_total_loss = saving_dict["d_loss"]
        g_loss = saving_dict["g_loss"]
        d_optimizer.load_state_dict(saving_dict["d_optimizer"])
        g_optimizer.load_state_dict(saving_dict["g_optimizer"])
        start_epoch = saving_dict["epoch"] + 1

    # generate the labels

    total_step = len(data_loader)
    for epoch in range(start_epoch, start_epoch + num_epochs):
        for i, real_images in enumerate(data_loader):
            if len(real_images) != batch_size:
                continue
            # real_images = real_images.view(batch_size, -1)

            real_labels = torch.ones(real_images.size()[0], 1)
            fake_labels = torch.zeros(real_images.size()[0], 1)

            ##################################################################################
            #                       Optimize the discriminator
            ##################################################################################

            # Feed the discriminator with the real images
            noise = torch.normal(0, 0.1, real_images.size()[1:])
            noisy_images = real_images + noise
            outputs = D(noisy_images)
            d_loss_real = criterian(outputs, real_labels)

            # feed the discriminator with fake images
            z = torch.randn(batch_size, noise_size)
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterian(outputs, fake_labels)

            # Sum up the losses from previous steps
            d_total_loss = d_loss_fake + d_loss_real
            reset_grad()
            d_total_loss.backward()
            d_optimizer.step()

            ##################################################################################
            #                       Optimize the generator
            ##################################################################################
            z = torch.randn(batch_size, noise_size)
            fake_images = G(z)
            outputs = D(fake_images)
            g_loss = criterian(outputs, real_labels)
            fake_results = fake_images

            reset_grad()
            g_loss.backward()
            g_optimizer.step()

            summary.add_scalar("discriminator loss", d_total_loss.item(), epoch)
            summary.add_scalar("generator loss", g_loss.item(), epoch)

            if i % int(total_step / 3) == 0:
                print("Epoch [{:4}/{}], "
                      "total_step: [{:3}/{}], "
                      "g_loss: {:.4f}, "
                      "d_loss:{:.4f}".format(epoch,
                                             num_epochs + start_epoch,
                                             i, total_step,
                                             g_loss.item(),
                                             d_total_loss.item()))

        # Show a grid of the images
        grid_size = 3
        numpy_images = fake_results.detach().numpy()
        some_random = np.random.randint(0, len(numpy_images), grid_size)
        summary.add_images("Some samples", numpy_images[some_random, :].reshape([grid_size, 1, 8, 8]), epoch)

        # show the generated fake images in the output
        # summary.add_image(tag="fake_image",
        #                   img_tensor=fake_results[0].view([1, 8, 8]),
        #                   global_step=epoch)

        # save the model
        saving_dict = {
            "epoch": epoch,
            "discriminator": D.state_dict(),
            "generator": G.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "d_loss": d_total_loss,
            "g_loss": g_loss,
        }

        # save the dictionary
        save_model(SAVE_DIR, saving_dict)
