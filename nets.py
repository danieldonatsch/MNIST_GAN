""" Generator and Discriminator classes.
Original script and code structure:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
Net architecture from:
https://pub.towardsai.net/building-training-gan-model-from-scratch-in-python-4cc718741332
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, version=0, len_z=10):
        super(Generator, self).__init__()
        self.main = None
        if version == 0:
            self.net_version0(len_z)
        else:
            print(f"No version {version} available for the Generator class")

    def net_version0(self, len_z: int) -> None:
        """Net Version 0, from the following Medium article (ported from Tensorflow to PyTorch)

        'Building & Training GAN Model From Scratch In Python' by Youssef Hosni
        https://pub.towardsai.net/building-training-gan-model-from-scratch-in-python-4cc718741332

        :param len_z:
        :return: None
        """
        print("Build Generator v0")
        self.main = nn.Sequential(
            # input vector with input_vector_len randon numbers plus
            # 10 numbers for the one-hot encoding of the requested digit
            nn.Linear(10+len_z, 7*7*128),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ReLU(True),
            # state size. ``128 x 7 x 7``
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. ``128 x 14 x 14``
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. ``64 x 28 x 28``
            nn.Conv2d(64, 1, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. ``1 x 28 x 28``
        )

    def forward(self, z: torch.tensor, exp_digit: torch.tensor) -> torch.tensor:
        """Forward pass of the Generator.

        Expects as input a vector z with random numbers and digit which is expected to be in the generated image.

        :param z: (torch.tensor) random number vector
        :param exp_digit: (torch.tensor) digit to be shown in the output image
        :return: (torch.tensor) generated image
        """
        # Convert exp. num into a one-hot encoding:
        one_hot = nn.functional.one_hot(exp_digit, num_classes=10)
        x = torch.concat((one_hot, z), dim=1)

        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, version=0):
        """Initialize the Discriminator

        :param version: Net version
        """
        super(Discriminator, self).__init__()
        self.main = None
        if version == 0:
            self.net_version0()
        else:
            print(f"No version {version} available for Discriminator class")

    def net_version0(self) -> None:
        """Discriminator (version 0), from the following Medium article (ported from Tensorflow to PyTorch):

        'Building & Training GAN Model From Scratch In Python' by Youssef Hosni
        https://pub.towardsai.net/building-training-gan-model-from-scratch-in-python-4cc718741332

        :return: None
        """
        print("Build Discriminator v0")
        self.main = nn.Sequential(
            # input size: ``1 x 28 x 28``
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # feature size ``32 x 14 x 14``
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # feature size. ``32 x 8 x 8``
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # feature size. ``128 x 4 x 4``
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # feature size. ``256 x 4 x 4``
            nn.Flatten(),
            nn.Linear(256*4*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass of the Discriminator net. Expects a 28x28 (MNIST-like) image as input.

        :param x:
        :return:
        """
        return self.main(x)
