"""Trains a Generator and a Discriminator convnet such that the Generator generates MNIST-like images from numbers
while the Disciriminator tries to figure out if any given image is made my the Generator or is from MNIST data set.
Faoolows the DCGAN tutorial from PyTorch website:
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
import torchvision.transforms as transf
import matplotlib.pyplot as plt

from nets import Generator, Discriminator
from train_classifier import ClassifierNet


def get_user_input():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--net-version', type=int, default=0, metavar='N',
                        help='Number of the net/model version (default: 0)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='Number of epochs to train (default: 14)')
    parser.add_argument('--glr', type=float, default=0.0002, metavar='LR',
                        help='learning rate for the generator (default: 0.0002)')
    parser.add_argument('--dlr', type=float, default=0.0002, metavar='LR',
                        help='learning rate for the discriminator (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, metavar='M',
                        help='Learning rate step beta1 (default: 0.5)')
    parser.add_argument('--delta-alpha', type=float, default=0.0, metavar='M',
                        help='The change of alpha after each epoch (default: 0.0)')
    parser.add_argument('--scheduler-step', type=int, default=100, metavar='N',
                        help='Number of epochs between learning rate changes (default: 100)')
    parser.add_argument('--scheduler-gamma', type=float, default=1.0, metavar='M',
                        help='Factor to change learning rate (default: 1.0, i.e. no change)')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='Disables GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Quickly check a single pass')
    parser.add_argument('--seed', type=int, default=999, metavar='N',
                        help='Random seed (default: 999)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='How many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--out-dir', type=str, default='out',
                        help="(Path to) output directory (default: 'out')")
    parser.add_argument('--len_z', type=int, default=20, metavar='N',
                        help='Length of the input vector z (with random numbers) to the generator. Default: 20')
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
    g = Generator(version=args.net_version, len_z=args.len_z).to(device)
    # Create the Discriminator
    d = Discriminator(version=args.net_version).to(device)
    # Create the Classifier, needed for the loss
    c = ClassifierNet().to(device)

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
    g.apply(weights_init)
    d.apply(weights_init)
    # Load existing classifier weights
    c.load_state_dict(torch.load('mnist_classifier_weights.pt'))
    c.eval()

    return g, d, c


def create_generator_input(batch_size, device, args):
    """Creates input for the generator.

    For each input a 2x7x7 tensor,
        one 7x7 "layer" with the number which should be drawn in the image,
        one 7x7-"layer" with random values.
    Total output has batch_size x 2 x 7 x 7, plus a list with the target numbers

    :param batch_size: (int) batch size
    :param device: (torch.device)
    :return: generator_input, expected numbers
    """
    return (torch.randn(batch_size, args.len_z, device=device),
            torch.randint(0, 10, size=(batch_size, ), device=device))


def main():
    """main() runs the main loop, including the actual training.

    :return:
    """

    # Get the user arguments
    args = get_user_input()
    print("User given arguments:")
    for k, v in args.__dict__.items():
        print(f"- {k}: {v}")

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    set_random_seed(args)

    device = get_device(args)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results

    # Get the data and the nets we need
    dataloader = get_dataloader(args)
    generator, discriminator, classifier = get_nets(device, args)

    # Initialize the ``BCELoss`` for the discriminator and the NLLLoss for the classifier
    disc_criterion = nn.BCELoss()
    class_criterion = nn.NLLLoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_gen_in, fixed_exp_nums = create_generator_input(1000, device, args)

    # Establish convention for real and gen_imgs labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both generator and discriminator
    opt_gen = optim.Adam(generator.parameters(), lr=args.glr, betas=(args.beta1, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=args.dlr, betas=(args.beta1, 0.999))
    scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=args.scheduler_step, gamma=args.scheduler_gamma, verbose=True)
    scheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=args.scheduler_step, gamma=args.scheduler_gamma, verbose=True)

    # Lists to keep track of progress
    losses_gen = []
    losses_disc = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(args.epochs):
        alpha = 0.5 + (epoch * args.delta_alpha)
        print("alpha:", alpha)

        generator.train()
        discriminator.train()

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
            err_disc_real = disc_criterion(output, label)
            # Calculate gradients for D in backward pass
            err_disc_real.backward()
            # Compute mean value for the discriminator on real MNIST images
            D_x += output.mean().item()

            # # Train with generated images batch
            gen_input, exp_nums = create_generator_input(b_size, device, args)
            # Generate image batch with G
            gen_imgs = generator(gen_input, exp_nums)
            label.fill_(fake_label)
            # Classify the generated images with the discriminator
            output = discriminator(gen_imgs.detach()).view(-1)
            # Calculate discriminator's loss on the all-gen_imgs batch
            err_disc_gen = disc_criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            err_disc_gen.backward()
            # Compute error of D as sum over the generated and the real images batches
            err_discriminator = err_disc_real + err_disc_gen
            # Update D
            if i % 2 == 0:
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
            err_generator = (1 - alpha) * disc_criterion(output_disc, label) + \
                            alpha * class_criterion(output_class, exp_nums)
            # Calculate gradients for G
            err_generator.backward()
            # Update G
            opt_gen.step()
            # Compute probabilities resp. accuracy for to display
            D_G_z2 += output_disc.mean().item()
            loss_gen += err_generator.item()
            gen_acc += exp_nums.eq(output_class.argmax(dim=1)).sum().item() / b_size

            # Output training stats
            if (i+1) % args.log_interval == 0 or args.dry_run:
                print(f"[{epoch+1}/{args.epochs}][{i+1}/{len(dataloader)}]" +
                      f"\tLoss_D: {loss_disc/args.log_interval:.4f}\tLoss_G: {loss_gen/args.log_interval:.4f}" +
                      f"\tD(x): {D_x/args.log_interval:.4f}\t" +
                      f"D(G(z)): {D_G_z1/args.log_interval:.4f} / {D_G_z2/args.log_interval:.4f}" +
                      f"\tC(G(z): {gen_acc/args.log_interval:.4%}")
                loss_disc, loss_gen, D_x, D_G_z1, D_G_z2, gen_acc = 0, 0, 0, 0, 0, 0

            # Save Losses for plotting later
            losses_gen.append(err_generator.item())
            losses_disc.append(err_discriminator.item())

            iters += 1

            if args.dry_run:
                break

        # End of Epoch:
        # Let the scheduler make a step
        scheduler_gen.step()
        scheduler_disc.step()
        # run once the fixed noise
        generator.eval()
        discriminator.eval()
        gen_imgs = generator(fixed_gen_in, fixed_exp_nums)
        output_class = classifier(gen_imgs)
        output_disc = discriminator(gen_imgs).view(-1)
        disc_error = output_disc.mean().item()
        derived_num = output_class.argmax(dim=1)
        correct = fixed_exp_nums.eq(derived_num).sum().item()
        accuracy = correct / fixed_gen_in.size(0)
        # Move them back to the CPU
        gen_imgs = gen_imgs.detach().cpu()
        derived_num = derived_num.detach().cpu()

        print(f"\nFixed Input: {epoch+1}/{args.epochs} epochs, discriminator error {disc_error:.4%}, " +
              f"number classifier accuracy: {accuracy:.4%} ({correct} / {fixed_gen_in.size(0)})\n")

        plt.figure(figsize=(32, 18))
        for sample_no in range(min(60, gen_imgs.size(0))):
            plt.subplot(6, 10, sample_no+1)
            plt.title(f"T: {int(fixed_exp_nums[sample_no].item())} (C: {derived_num[sample_no]})",
                      {'color': 'g' if fixed_exp_nums[sample_no].item() == derived_num[sample_no] else 'r'})
            plt.imshow(gen_imgs[sample_no, 0, :, :].cpu().detach().numpy(), cmap='gray')
            border_color = 'green' if output_disc[sample_no] > 0.499 else 'red'
            for side in ['top', 'right', 'bottom', 'left']:
                plt.gca().spines[side].set_color(border_color)
            plt.xticks([])
            plt.yticks([])

        plt.savefig(os.path.join(out_dir, f"fixed_input_samples_epoch={epoch+1}_acc={int(100*accuracy):02d}.png"))

        if epoch % 10 == 0 or (epoch + 1) == args.epochs:
            if args.save_model:
                torch.save(generator.state_dict(), os.path.join(out_dir, f"generator_weights_{epoch:03d}.pt"))
                torch.save(discriminator.state_dict(), os.path.join(out_dir, f"discriminator_weights_{epoch:03d}.pt"))
            plt.figure(figsize=(32, 18))
            plt.title("Loss Development")
            plt.plot(losses_gen, label='Generator loss')
            plt.plot(losses_disc, label='Discriminator loss')
            plt.legend()
            plt.savefig(os.path.join(out_dir, f"loss_development.png"))
            plt.show()
        else:
            plt.close()

        if args.dry_run:
            break

    # Save the models - in any case!
    torch.save(generator.state_dict(), os.path.join(out_dir, "generator_weights_final.pt"))
    torch.save(discriminator.state_dict(), os.path.join(out_dir, "discriminator_weights_final.pt"))


if __name__ == '__main__':
    main()
