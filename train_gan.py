"""
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torchvision import datasets, transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from nets import Generator, Discriminator


# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


def get_user_input():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=999, metavar='S',
                        help='random seed (default: 999)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    return parser.parse_args()


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_dataloader(args) -> torch.utils.data.DataLoader:
    """ Assembles the data loader

    :param args: args from ArgumentParser
    :return: torch.utils.data.DataLoader object
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)


def get_device(args) -> torch.device:
    """Checks if user requested GPU or CPU training and if GPUs are available. Also deals with processors architecture.

    :param args: args from ArgumentParser
    :return: torch.device
    """
    if args.no_gpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("GPU-training not possible")
    return torch.device("cpu")


def set_random_seed(args):
    """ Set random seed for reproducibility

    :param args: args from ArgumentParser
    :return: None
    """
    if args.seed > 0:
        seed = args.seed
    else:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_nets(device, args) -> tuple:
    """Sets up the generator and discriminator nets

    :param device: torch.device
    :param args: args from ArgumentParser
    :return: tuple(generator net, discriminator net)
    """
    # Create the generator
    ngpu = 1    # TODO make it an argument
    g = Generator(ngpu, 1).to(device)
    # Create the Discriminator
    d = Discriminator(ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        g = nn.DataParallel(g, list(range(ngpu)))
        d = nn.DataParallel(d, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
    g.apply(weights_init)
    d.apply(weights_init)

    # Print the model
    #print(g)
    #print(d)

    return g, d


def train_discriminator():
    pass


def train_generator():
    pass


def main():

    args = get_user_input()

    set_random_seed(args)

    device = get_device(args)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results

    dataloader = get_dataloader(args)

    generator, discriminator = get_nets(device, args)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, 1, 7, 7, device=device)

    # Establish convention for real and gen_imgs labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both generator and discriminator
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    losses_gen = []
    losses_disc = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_imgs = data[0].to(device)
            b_size = real_imgs.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_imgs).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with generated images batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, 1, 7, 7, device=device)
            # Generate gen_imgs image batch with G
            gen_imgs = generator(noise)
            label.fill_(fake_label)
            # Classify all gen_imgs batch with the discriminator
            output = discriminator(gen_imgs.detach()).view(-1)
            # Calculate discriminator's loss on the all-gen_imgs batch
            errD_gen = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_gen.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the generated and the real images batches
            errD = errD_real + errD_gen
            # Update D
            opt_disc.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # gen_imgs labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-gen_imgs batch through D
            output = discriminator(gen_imgs).view(-1)
            # Calculate generator's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            opt_gen.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            losses_gen.append(errG.item())
            losses_disc.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    gen_imgs = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(gen_imgs, padding=2, normalize=True))

            iters += 1

        plt.figure()
        for i in range(min(24, gen_imgs.size(0))):
            plt.subplot(4, 6, i+1)
            plt.imshow(gen_imgs[i, 0, :, :].cpu().detach().numpy(), cmap='gray')

        plt.figure()
        plt.title("Loss Development")
        plt.plot(losses_gen, label='Generator loss')
        plt.plot(losses_disc, label='Discriminator loss')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
