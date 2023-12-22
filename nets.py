# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, input_channels, version=0):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = None
        if version == 1:
            self.net_version1(input_channels)
        elif version == 2:
            self.net_version2(input_channels)
        else:
            self.net_version0(input_channels)

    def net_version0(self, input_channels) -> None:
        """First version... coming from the PyTorch website's sample

        :param input_channels:
        :return: None
        """
        print("Build Generator v0")
        self.main = nn.Sequential(
            # input is Z, 7x7, going into a convolution
            nn.Conv2d(input_channels, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # state size. ``8 x 7 x 7``
            nn.Conv2d(8, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # state size. ``16 x 7 x 7``
            nn.ConvTranspose2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. ``32 x 14 x 14``
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. ``32 x 14 x 14``
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``1 x 28 x 28``
        )

    def net_version1(self, input_channels) -> None:
        """Net Version 1, tries to make the original version somewhat smaller

        :param input_channels:
        :return: None
        """
        print("Build Generator v1")
        self.main = nn.Sequential(
            # input is Z, 7x7, going into a convolution
            nn.Conv2d(input_channels, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``8 x 7 x 7``
            nn.Conv2d(8, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``16 x 7 x 7``
            nn.ConvTranspose2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``32 x 14 x 14``
            #nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            #nn.BatchNorm2d(32),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. ``32 x 14 x 14``
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``1 x 28 x 28``
        )

    def net_version2(self, input_channels) -> None:
        """Net Version 2, from the following Medium article (ported from Tensorflow to PyTorch)

        “Building & Training GAN Model From Scratch In Python“ by Youssef Hosni
        https://pub.towardsai.net/building-training-gan-model-from-scratch-in-python-4cc718741332

        :param input_channels:
        :return: None
        """
        print("Build Generator v2")
        self.main = nn.Sequential(
            # input vector with randon numbers of length 100
            nn.Linear(20, 7*7*128),
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

    def forward(self, input, exp_num):
        # Convert exp. num into a one-hot encoding:
        one_hot = nn.functional.one_hot(exp_num, num_classes=10)
        x = torch.concat((one_hot, input), dim=1)
        '''
        print("Generator Layers:")
        for layer in self.main:
            x = layer(x)
            print("  ", x.size())
        return x
        '''
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ngpu, version=0):
        """

        :param version:
        :param ngpu:
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ngpu = ngpu
        self.main = None
        # No specific version 1...
        if version == 2:
            self.net_version2()
        else:
            self.net_version0()

    def net_version0(self) -> None:
        """First version... coming from the PyTorch website's sample

        :param input_channels:
        :return: None
        """
        print("Build Discriminator v0")
        self.main = nn.Sequential(
            # input is ``1 x 28 x 28``
            nn.Conv2d(1, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``16 x 14 x 14``
            #nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(32),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. ``32 x 14 x 14``
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``32 x 7 x 7``
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 3 x 3`` ???
            nn.Conv2d(64, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def net_version2(self) -> None:
        """Discriminator Version 2, from the following Medium article (ported from Tensorflow to PyTorch):

        “Building & Training GAN Model From Scratch In Python“ by Youssef Hosni
        https://pub.towardsai.net/building-training-gan-model-from-scratch-in-python-4cc718741332

        :return: None
        """
        print("Build Discriminator v2")
        self.main = nn.Sequential(
            # input is ``1 x 28 x 28``
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``32 x 14 x 14``
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``32 x 8 x 8``
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``128 x 4 x 4``
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``256 x 4 x 4``
            nn.Flatten(),
            nn.Linear(256*4*4, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        '''
        print("Discriminator layer sizes:")
        x = input
        for layer in self.main:
            x = layer(x)
            print("  ", x.size())
        return x
        '''
        return self.main(input)
