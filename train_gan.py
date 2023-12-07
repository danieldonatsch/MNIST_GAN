"""
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transf
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from nets import Generator, Discriminator
from train_classifier import ClassifierNet


def get_user_input():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, metavar='M',
                        help='Learning rate step beta1 (default: 0.5)')
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
    transform = transf.Compose([
        transf.ToTensor(),
        transf.Normalize((0.5,), (0.5,))
        ])
    dataset = dset.MNIST('../data', train=True, download=True, transform=transform)
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
    g = Generator(ngpu, 2).to(device)
    # Create the Discriminator
    d = Discriminator(ngpu).to(device)
    # Create the Classifier, needed for the loss
    c = ClassifierNet().to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        g = nn.DataParallel(g, list(range(ngpu)))
        d = nn.DataParallel(d, list(range(ngpu)))
        c = nn.DataParallel(c, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
    g.apply(weights_init)
    d.apply(weights_init)
    # Load existing classifier weights
    c.load_state_dict(torch.load('mnist_classifier_acc_9917-10000.pt'))
    c.eval()

    # Print the model
    #print(g)
    #print(d)
    #print(c)

    return g, d, c


def create_generator_input(batch_size, device):
    """Creates input for the generator.

    For each input a 2x7x7 tensor,
        one 7x7 "layer" with the number which should be darwn in the image,
        one 7x7-"layer" with random values.
    Total output has batch_size x 2 x 7 x 7, plus a list with the target numbers

    :param batch_size: int, batch size
    :param device: torch.device
    :return: generator_input, expected numbers
    """
    # Generate batch of generator input, consisting of noise and the target number
    noise = torch.randn(batch_size, 1, 7, 7, device=device)
    target_nums = torch.randint(0, 10, size=(batch_size, 1, 1, 1), device=device)
    req_num = (target_nums / 4.5) - 1.0
    req_num = torch.repeat_interleave(req_num, 7, 2)
    req_num = torch.repeat_interleave(req_num, 7, 3)
    generator_input = torch.cat((req_num, noise), dim=1)

    return generator_input, target_nums.view(-1)


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

    generator, discriminator, classifier = get_nets(device, args)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_gen_in, fixed_exp_nums = create_generator_input(64, device)

    # Establish convention for real and gen_imgs labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both generator and discriminator
    opt_disc = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    losses_gen = []
    losses_disc = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(args.epochs):
        alpha = 0.5 + 0.5 * epoch/args.epochs  # The weight of the number classifier
        print("alpha:", alpha)
        loss_disc, loss_gen, D_x, D_G_z1, D_G_z2, gen_acc = 0, 0, 0, 0, 0, 0
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # # Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_imgs = data[0].to(device)
            b_size = real_imgs.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_imgs).view(-1)
            # Calculate loss on all-real batch
            err_disc_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            err_disc_real.backward()
            # Compute mean value for the discriminator on real MNIST images
            D_x += output.mean().item()

            # # Train with generated images batch
            gen_input, exp_nums = create_generator_input(b_size, device)
            # Generate image batch with G
            gen_imgs = generator(gen_input)
            label.fill_(fake_label)
            # Classify the generated images with the discriminator
            output = discriminator(gen_imgs.detach()).view(-1)
            # Calculate discriminator's loss on the all-gen_imgs batch
            err_disc_gen = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            err_disc_gen.backward()
            # Compute error of D as sum over the generated and the real images batches
            err_discriminator = err_disc_real + err_disc_gen
            # Update D
            opt_disc.step()
            # Compute mean value for the discriminator on generated images
            D_G_z1 += output.mean().item()
            loss_disc += err_discriminator.item()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            classifier.zero_grad()
            label.fill_(real_label)  # The labels are real for generator cost (we try to be as real as possible!)
            # Since we just updated D, perform another forward pass of the generated image batch through D
            output_disc = discriminator(gen_imgs).view(-1)
            # Also, infer the number the represent (or: they are supposed to...)
            output_class = classifier(gen_imgs)
            # Calculate generator's loss based on this output
            err_generator = (1 - alpha) * criterion(output_disc, label) + \
                            alpha * nn.functional.nll_loss(output_class, exp_nums)
            # Calculate gradients for G
            err_generator.backward()
            # Update G
            opt_gen.step()
            # Compute probabilities resp. accuracy for to display
            D_G_z2 += output_disc.mean().item()
            loss_gen += err_generator.item()
            gen_acc += exp_nums.eq(output_class.argmax(dim=1)).sum().item() / b_size

            # Output training stats
            freq = 50
            if (i+1) % freq == 0:
                print(f"[{epoch+1}/{args.epochs}][{i+1}/{len(dataloader)}]" +
                      f"\tLoss_D: {loss_disc/freq:.4f}\tLoss_G: {loss_gen/freq:.4f}" +
                      f"\tD(x): {D_x/freq:.4f}\tD(G(z)): {D_G_z1/freq:.4f} / {D_G_z2/freq:.4f}" +
                      f"\tC(G(z): {gen_acc/freq:.4%}")
                loss_disc, loss_gen, D_x, D_G_z1, D_G_z2, gen_acc = 0, 0, 0, 0, 0, 0

            # Save Losses for plotting later
            losses_gen.append(err_generator.item())
            losses_disc.append(err_discriminator.item())

            iters += 1

        # End of Epoch, run once the fixed noise
        gen_imgs = generator(fixed_gen_in)
        output_class = classifier(gen_imgs)
        derived_num = output_class.argmax(dim=1)
        correct = fixed_exp_nums.eq(derived_num).sum().item()
        accuracy = correct / fixed_gen_in.size(0)
        # Move them back to the CPU
        gen_imgs = gen_imgs.detach().cpu()
        derived_num = derived_num.detach().cpu()
        img_list.append(vutils.make_grid(gen_imgs, padding=2, normalize=True))

        print(f"\nFixed Input: {epoch+1}/{args.epochs} epochs, " +
              f"number classifier accuracy: {accuracy:.4%} ({correct} / {fixed_gen_in.size(0)})\n")

        plt.figure()
        for sample_no in range(min(60, gen_imgs.size(0))):
            plt.subplot(6, 10, sample_no+1)
            plt.title(f"{int(fixed_exp_nums[sample_no].item())} ({derived_num[sample_no]})")
            plt.imshow(gen_imgs[sample_no, 0, :, :].cpu().detach().numpy(), cmap='gray')

        plt.savefig(f"fixed_input_samples_epoch={epoch+1}_acc={int(100*gen_acc):02d}.png")

        if epoch % 2 == 0:
            plt.figure()
            plt.title("Loss Development")
            plt.plot(losses_gen, label='Generator loss')
            plt.plot(losses_disc, label='Discriminator loss')
            plt.legend()
            plt.show()
        else:
            plt.close()


if __name__ == '__main__':
    main()
