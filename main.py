#
# Adapted from code 'main.py' for the paper 'Invariant Risk Minimization'
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted by Kejie Zhao on Jan 6 2024
#

import argparse
import numpy as np
import torch
import pathlib
from tqdm import tqdm
from torch import nn, optim, autograd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.MLP import *
from models.CNN import *
from OddMNIST import *

parser = argparse.ArgumentParser(description='OddMNIST')
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=10000)
parser.add_argument('--grayscale_model', action='store_true', default=True)
parser.add_argument('--model', type=str, default='mlp')
parser.add_argument('--dataset_root', type=str, default='./processed_data/')
parser.add_argument('--seed', type=int, default=20240106)
flags = parser.parse_args()

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

dataset_root = pathlib.Path(flags.dataset_root)
train_root = dataset_root.joinpath('train')
val_root = dataset_root.joinpath('val')
test_root = dataset_root.joinpath('test')

final_train_accs = []
final_test_accs = []

# Load OddMNIST, make train/val splits, and shuffle train set examples

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_dataset = OddMNIST(root=train_root, train=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_dataset = OddMNIST(root=val_root, train=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)
mnist_train = (train_dataset.data[:60000], train_dataset.targets[:60000])
mnist_val = (val_dataset.data[:100], val_dataset.targets[:100])

if flags.model == 'mlp':
    model = MLP().to(device)
# elif flags.model == 'cnn':
#     model = CNN().to(device)
else:
    raise NotImplementedError

# Set torch seed
torch.manual_seed(flags.seed)
rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())


# Build environments
def make_environment(images, labels):
    return {
        'images': images.float().to(device),
        'labels': labels[:, None].to(device)
    }


envs = [
    make_environment(mnist_train[0], mnist_train[1]),
    make_environment(mnist_val[0], mnist_val[1]),
]


# Define loss function helpers
# Changed this to calculate multi-class loss
def mean_nll(logits, y):
    #  return nn.functional.binary_cross_entropy_with_logits(logits, y)
    probs = nn.functional.softmax(logits, dim=1)
    # print(logits.shape, y.shape)
    # print(logits.shape, torch.squeeze(y).shape)
    # return nn.functional.cross_entropy(logits, y)
    return nn.functional.cross_entropy(probs, torch.squeeze(y))


# Changed this to calculate multi-class accuracy
def mean_accuracy(logits, y):
    probs = nn.functional.softmax(logits, dim=1)
    _, preds = torch.max(probs, dim=1)
    # preds = (logits > 0.).float()
    # return ((preds - y).abs() < 1e-2).float().mean()
    correct = (preds == torch.squeeze(y)).sum().item()
    return torch.tensor((correct / len(preds)))


def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


# Train loop

def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


optimizer = optim.Adam(model.parameters(), lr=flags.lr)

pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')


def main():
    # for step in range(flags.steps):
    # record loss and accuracy
    loss_hist = []
    penalized_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    for step in tqdm(range(flags.steps)):
        for env in envs:
            logits = model(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
        train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

        train_acc_hist.append(train_acc.detach().cpu().numpy())
        loss_hist.append(train_nll.detach().cpu().numpy())

        weight_norm = torch.tensor(0.).cuda()
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        penalty_weight = (flags.penalty_weight
                          if step >= flags.penalty_anneal_iters else 1.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        penalized_loss_hist.append(loss.detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_acc = envs[1]['acc']
        val_acc_hist.append(val_acc.detach().cpu().numpy())

        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                val_acc.detach().cpu().numpy()
            )

        final_train_accs.append(train_acc.detach().cpu().numpy())
        final_test_accs.append(val_acc.detach().cpu().numpy())
        print('\nFinal train acc (mean/std):')
        print(np.mean(final_train_accs), np.std(final_train_accs))
        print('Final validation acc (mean/std):')
        print(np.mean(final_test_accs), np.std(final_test_accs))

    # Plot loss and accuracy
    plt.plot(loss_hist, 'b-', label='Training Loss')
    plt.plot(penalized_loss_hist, 'y-', label='Penalized Training Loss')
    plt.plot(train_acc_hist, 'r-', label='Training Accuracy')
    plt.plot(val_acc_hist, 'g-', label='Validation Accuracy')
    plt.title('Loss and Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()
    # Save model parameters
    torch.save(model.state_dict(), './saved_models/{}_{}_{}.pth'.format(flags.model, flags.steps, flags.seed))


if __name__ == "__main__":
    main()
