# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, input_channels):
        super(Generator, self).__init__()
        self.ngpu = ngpu
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

    def forward(self, input):
        '''
        print("Here!")
        x = input
        for layer in self.main:
            x = layer(x)
            print(x.size())
        return x
        '''
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``1 x 28 x 28``
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``16 x 28 x 28``
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``32 x 14 x 14``
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``64 x 7 x 7``
            nn.Conv2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 3 x 3`` ???
            nn.Conv2d(64, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        '''
        x = input
        for layer in self.main:
            x = layer(x)
            print(x.size())
        return x
        '''
        return self.main(input)
