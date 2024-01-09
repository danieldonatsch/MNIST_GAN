# MNIST-Like Number Generator

Goal of the scripts in this repo was to train a GAN 
and get experience with this type of machine learning problems.
The set goal is to generate MNIST-like images for a given digit.
If you want to play with these scripts, follow the instructions.

## Installation

Checkout the repository and build an environment.
The modules and packages used you can find in environment.yaml.
In you're a conda user, run

    conda env create -f environment.yaml

## Classifier

In the first step I trained an MNIST digits classifier.
To start with I used example code from the [PyTorch website](https://pytorch.org/examples/).
I had to learn, that ReLu does not back-propagate the loss well enough,
to teach the generator.
Therefore, I changed the ReLu layers to LeakyReLu.
Beyond that, I left the script mostly untouched.
The trained weights are saved in `mnist_classifier_weights.pt` and are part of this repo.
If you care just about the GAN, use the pretrained weights directly.
If you want to train your own classifier weights run

    train_classifier.py --save-model

With the `-h` option you get all possible scirpt parameters.

## Actual GAN-Training

The two nets trained in a GAN, the generator and the discriminator are in `nets.py`.
The script to train them is obviously the `train_gan.py`.
The script needs classifier weights.
They need to be placed next to the script (same folder) and be named `mnist_classifier_weights.pt`.
Once this is set up, you're ready to play.
Start with the following command to get all parameter options:

    train_gan.py -h

## Thoughts and Lessons Learnt

* ReLu does not do a good job in backpropagating information. See also classifier section.
* Knowing, when the creator is well-trained, is really difficult. :-)