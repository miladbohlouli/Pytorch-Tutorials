import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import numpy as np
from torchvision.transforms import ToTensor
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

summary_writer_general = SummaryWriter("./logs/")
summary_writer_generator = SummaryWriter("./logs/discriminator/")
summary_writer_discriminator = SummaryWriter("./logs/generator/")

parser = argparse.ArgumentParser()
# General parameters
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--image_size", default=28*28, type=int)
parser.add_argument("--noise_size", default=100, type=int)
parser.add_argument("--iterations", default=10, type=int)
parser.add_argument("--use_gpu", default=False, type=bool)
parser.add_argument("--model_dir", default="./save", type=str)
parser.add_argument("--loading_strategy", default="last", type=str)
parser.add_argument("--save_every_d_epochs", default=3, type=int)
parser.add_argument("--ignore_first_iterations", default=15, type=int)

# Generator parameters
parser.add_argument("--generator_structure", default=[100, 256, 256, 28*28], type=list)
parser.add_argument("--generator_activation", default="Relu", type=str)
parser.add_argument("--generator_dropout", default=0.0, type=float)
parser.add_argument("--g_steps", default=1, type=int)
parser.add_argument("--g_lr", default=0.002, type=float)

# Discriminator parameters
parser.add_argument("--disc_structure", default=[28*28, 256, 128], type=list)
parser.add_argument("--disc_activation", default="Relu", type=str)
parser.add_argument("--disc_dropout", default=0.0, type=float)
parser.add_argument("--d_steps", default=1, type=int)
parser.add_argument("--d_lr", default=0.0002, type=float)

def make_mlp(layers: list, activation:str = "Relu", dropout:float = 0.0):
    nn_layers = []
    for dim_in, dim_out in zip(layers[:-1], layers[1:]):
        nn_layers.append(nn.Linear(dim_in, dim_out))
        if activation == "Relu":
            nn_layers.append(nn.ReLU())
        elif activation == "LeakyRelu":
            nn_layers.append(nn.LeakyReLU())
        if dropout > 0:
            nn_layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*nn_layers)

def init_weights(m):
    if m.__class__.__name__ == "Linear":
        nn.init.kaiming_uniform(m.weight)

def bce_loss(input, target):
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def get_tensor_type(args):
    if args.use_gpu:
        return torch.cuda.FloatTensor
    else:
        return torch.FloatTensor

def checkpoint_path(args):
    try:
        directories = os.listdir(args.model_dir)
    except:
        os.makedirs(args.model_dir)
        return None
    processed_path = dict()
    processed_path["checkpoint"] = []

    for dir in directories:
        if "best" in dir:
            processed_path["best"] = dir

        elif "checkpoint" in dir:
            processed_path["checkpoint"].append(tuple((dir.split(sep="-")[1].split(sep=".")[0], dir)))

    if len(processed_path["checkpoint"]) == 0 and "best" not in processed_path:
        return None

    elif args.loading_strategy == "best":
        return os.path.join(args.model_dir, processed_path["best"])

    elif args.loading_strategy == "last":

        # Finding the last saved_checkpoint
        directories_list = np.asarray(processed_path["checkpoint"], dtype=object)
        return os.path.join(args.model_dir, directories_list[directories_list[:, 0].astype(np.int).argmax(), 1])

# define the model
class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        generator_structure = args.generator_structure
        assert generator_structure[0] == args.noise_size
        assert  generator_structure[-1] == args.image_size

        self.generator_model = make_mlp(layers=generator_structure,
                                        activation=args.generator_activation,
                                        dropout=args.generator_dropout)

    def forward(self, noise):
        return self.generator_model(noise)


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        discriminator_structure = args.disc_structure
        assert discriminator_structure[0] == args.image_size
        self.discriminator_model = make_mlp(layers=discriminator_structure,
                                            activation=args.disc_activation,
                                            dropout=args.disc_dropout)
        self.linear_out = nn.Linear(discriminator_structure[-1], 1)

    def forward(self, images):
        out = self.discriminator_model(images)
        logits = self.linear_out(out)
        return logits


def main(args):

    dataset_train = MNIST(root="../datasets/",
                          train=True,
                          transform=ToTensor())

    dataset_test = MNIST(root="../datasets/",
                         train=False,
                         transform=ToTensor())

    train_loader = DataLoader(dataset_train,
                              batch_size=args.batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset_test,
                             batch_size=args.batch_size,
                             shuffle=True)

    # Construct the models
    logger.info("Here is the generator")
    g = Generator(args)
    logger.info(g)

    logger.info("Here is the discriminator")
    d = Discriminator(args)
    logger.info(d)

    # Initilize the weights
    g.apply(init_weights)
    d.apply(init_weights)

    # Transfer the tensors to GPU if required
    tensor_type = get_tensor_type(args)
    g.type(tensor_type).train()
    d.type(tensor_type).train()

    # defining the loss and optimizers for generator and discriminator
    d_optimizer = torch.optim.Adam(d.parameters(), lr=args.d_lr)
    g_optimizer = torch.optim.Adam(g.parameters(), lr=args.g_lr)


    # Loading the checkpoint if existing
    #   the loading strategy is based on the best accuracy and after every iteration interval

    loading_path = checkpoint_path(args)
    if loading_path is not None:
        logger.info(f"Loading the model in {loading_path}...")
        loaded_dictionary = torch.load(loading_path)
        g.load_state_dict(loaded_dictionary["generator"])
        d.load_state_dict(loaded_dictionary["discriminator"])
        g_optimizer.load_state_dict(loaded_dictionary["g_optimizer"])
        d_optimizer.load_state_dict(loaded_dictionary["d_optimizer"])
        start_epoch = loaded_dictionary["epoch"] + 1
        step = loaded_dictionary["step"]
        best_validation_loss = loaded_dictionary["best_validation_loss"]
        total_validation_loss = loaded_dictionary["total_validation_loss"]
        g_loss = loaded_dictionary["current_g_loss"]
        d_loss = loaded_dictionary["current_d_loss"]
        logger.debug(f"Done loading the model in {loading_path}")

    else:
        logger.info(f"No saved checkpoint, Initializing...")
        step = 0
        start_epoch = 0
        best_validation_loss = np.inf
        total_validation_loss = 0
        d_loss = np.inf
        g_loss = np.inf

    logger.debug("Training the model")
    for i in range(start_epoch, start_epoch + args.iterations):
        g.train()
        d.train()
        for x_train, _ in train_loader:
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            true_labels = torch.ones(x_train.shape[0], 1)
            fake_labels = torch.zeros(x_train.shape[0], 1)

            while d_steps_left > 0:
                ###################################################################
                #                 training the discriminator
                ###################################################################
                logger.debug("Training the discriminator")

                noise_tensor = torch.normal(0.0, 1.0, (x_train.shape[0], args.noise_size))

                x_train = x_train.reshape(-1, args.image_size)
                real_predictions = d(x_train)
                real_loss = bce_loss(real_predictions, true_labels)

                fake_images = g(noise_tensor)
                fake_prediction = d(fake_images)
                fake_loss = bce_loss(fake_prediction, fake_labels)

                d_loss = fake_loss + real_loss

                summary_writer_discriminator.add_scalar("Loss", d_loss, i)

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                d_steps_left -= 1

            while g_steps_left > 0:
                ###################################################################
                #                 training the generator
                ###################################################################
                logger.debug("Training the generator")
                noise_tensor = torch.normal(0.0, 1.0, (x_train.shape[0], args.noise_size))
                fake_images = g(noise_tensor)
                fake_prediction = d(fake_images)

                g_loss = bce_loss(fake_prediction, true_labels)

                summary_writer_generator.add_scalar("Loss", g_loss, i)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                g_steps_left -= 1

            step += 1

        if args.iterations > 0:
            logger.info(f"TRAINING[{i + 1}/{start_epoch + args.iterations}]\td_loss:{d_loss:.2f}\tg_loss:{g_loss:.2f}")
            summary_writer_general.add_images(f"samples_images_iteration_{i}",
                                              fake_images[np.random.randint(0, x_train.shape[0], 5)].squeeze(1).
                                              reshape([-1, 1, int(args.image_size ** 0.5), int(args.image_size ** 0.5)]))



        with torch.no_grad():
            logger.debug("Evaluating the model")
            g.eval()
            d.eval()

            for x_test, _ in test_loader:

                # Discriminator part
                noise_tensor = torch.normal(0.0, 1.0, (x_train.shape[0], args.noise_size))
                fake_labels = torch.zeros(x_train.shape[0])
                true_labels = torch.ones(x_train.shape[0])

                generated_images = g(noise_tensor)
                fake_scores = d(generated_images)
                fake_loss = bce_loss(fake_scores, fake_labels)

                x_test = x_test.reshape(-1, args.image_size)
                true_scores = d(x_test)
                true_loss = bce_loss(true_scores, true_labels)

                validation_d_loss = true_loss + fake_loss

                # Generator part
                noise_tensor = torch.normal(0.0, 1.0, (x_train.shape[0], args.noise_size))
                generated_images = g(noise_tensor)
                fake_scores = d(generated_images)

                validation_g_loss = bce_loss(fake_scores, true_labels)

                # Combine the losses
                total_validation_loss = validation_d_loss + validation_g_loss

            logger.info(f"VALIDATING\ttotal evaluation loss:"
                        f"{total_validation_loss:.2f}\tg_loss:{validation_g_loss:.2f}\td_loss:{validation_d_loss:.2f}")
            summary_writer_general.add_scalar("total_evaluation_loss", total_validation_loss, i)

        total_validation_loss = total_validation_loss.item()
        # check if it is time to save a checkpoint of the model
        if total_validation_loss <= best_validation_loss or \
                (i + 1) % args.save_every_d_epochs == 0 or \
                (i + 1) == start_epoch + args.iterations:
            logger.info("Saving the model....")
            checkpoint = {
                "epoch": i,
                "step": step,
                "generator": g.state_dict(),
                "discriminator": d.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
                "best_validation_loss": best_validation_loss,
                "current_d_loss": d_loss,
                "current_g_loss": g_loss,
                "total_validation_loss": total_validation_loss
            }
            if total_validation_loss <= best_validation_loss and i > args.ignore_first_iterations:
                best_validation_loss = total_validation_loss
                torch.save(checkpoint, args.model_dir + "/best.pt")

            if (i + 1) % args.save_every_d_epochs == 0 or (i + 1) == start_epoch + args.iterations:
                torch.save(checkpoint, args.model_dir + "/checkpoint-" + str(i + 1) + ".pt")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
