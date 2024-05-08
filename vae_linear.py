"""Code from here: https://avandekleut.github.io/vae/

Good video to understand KL-Divergence
https://www.youtube.com/watch?v=q0AkK8aYbLY


"""
import argparse
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt;


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        device = get_device()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        # Next lines make the encoder a variational one.
        # Instead of one layer for the embeding, we have two
        # One that creates mu, the other sigma.
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        # According to the linked StackExchange article,
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # we have:
        # KL = log( sigma2 / sigma1) + (sigma1^2 + (mu1-mu2)^2) / 2*sigma2^2 - 1/2
        # with sigma2 = 1, mu2 = 0 and sigma1 = sigma, mu1 = mu
        # we get:
        # KL = log( 1 / sigma ) + (sigma^2 + mu1^2) / 2 - 1/2
        # with         ^                              ^
        #              = -log(sigma)                  ???
        # is then (almost):
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, data, epochs=20):
    device = get_device()
    print(f"Start training for {epochs} epochs on {device}")
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        tot_loss = 0
        for x, y in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            tot_loss += loss.item()
            loss.backward()
            opt.step()
        print(f"  Epoch {epoch+1:2d}. Loss: {tot_loss/(len(data)*args.batch_size):.4f}")
    return autoencoder


def get_device():
    # If GPU usage is disabled, return just CPU
    if args.no_gpu:
        return torch.device("cpu")

    # Figure out, if we have an Apple (MPS) or NVidia (cuda) GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # If no GPU found, only CPU is left...
    return torch.device("cpu")


def plot_latent(autoencoder, data, num_batches=100):
    device = get_device()
    plt.figure()
    plt.title("Latent Vector Representations")
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break


def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    device = get_device()
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.figure()
    plt.title("Reconstructed Digits from Latent Space")
    plt.imshow(img, extent=[*r0, *r1])


def get_user_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST AutoEncoder Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--latent-dims', type=int, default=2, metavar='N',
                        help='size of latent space (default: 2)')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables training on GPU')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pretrained-model', type=str, required=False,
                        help='Path to the a .pt file with the pre-trained model weights.')

    return parser.parse_args()


def main():
    device = get_device()

    torch.manual_seed(args.seed)

    data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data',
                                   transform=torchvision.transforms.ToTensor(),
                                   download=True,
                                   train=True),
        batch_size=args.batch_size,
        shuffle=True)

    vae = VariationalAutoencoder(args.latent_dims).to(device)  # GPU

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        vae.load_state_dict(torch.load(args.pretrained_model))
    else:
        vae = train(vae, data)
        torch.save(vae.state_dict(), "vae_linear.pt")

    plot_latent(vae, data)
    plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
    plt.show()

if __name__ == '__main__':

    args = get_user_args()
    main()
