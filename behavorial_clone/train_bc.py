
from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

import models.mineNet


# Training settings
parser = argparse.ArgumentParser(description='Behavioural Cloning')
# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, metavar='M',
                    help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--model',
                    choices=['convnet', 'mymodel', 'mineNet'],
                    help='which model to train/evaluate')
parser.add_argument('--hidden-dim', type=int,
                    help='number of hidden features/activations')
parser.add_argument('--kernel-size', type=int,
                    help='size of convolution kernels/filters')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='number of batches between logging train status')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# Minecraft Data
n_classes = 256
im_size = (3, 64, 64)


print('Loading data...')

data = torch.load('image_data.pt').long()
labels = torch.load('labels_data.pt').long()
data = torch.transpose(data, 1, 3)

val = data[-1000:, :, :, :]
val_labels = labels[-1000:]

test = data[-1000:, :, :, :]
test_labels = labels[-1000:]

data = data[:-1000, :, : , :]
labels = labels[:-1000]



# DataLoaders
data_length = data.shape[0]
val_length = val.shape[0]
test_length = test.shape[0]

train_loader = []
val_loader = []
test_loader = []


for i in range(0, data_length, args.batch_size):
    train_loader.append({'data': data[i:(i+args.batch_size), :, :, :], 'labels':labels[i:(i+args.batch_size)]})
    
for i in range(0, val_length, args.batch_size):
    val_loader.append({'data': val[i:(i+args.batch_size), :, :, :], 'labels':val_labels[i:(i+args.batch_size)]})
    
for i in range(0, test_length, args.batch_size):
    test_loader.append({'data': test[i:(i+args.batch_size), :, :, :], 'labels':test_labels[i:(i+args.batch_size)]})
print('Data Loaded')


# Load the model
if args.model == 'mineNet':
    model = models.mineNet.CNN(im_size, args.hidden_dim, args.kernel_size,
                               n_classes)
    
# cross-entropy loss function
criterion = F.cross_entropy
if args.cuda:
    model.cuda()


opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


def train(epoch):

    model.train()

    
    for batch_idx, batch in enumerate(train_loader):
        # prepare data
        
        images, targets = Variable(batch['data']), Variable(batch['labels'])
        if args.cuda:
            images, targets = images.cuda(), targets.cuda()

        opt.zero_grad()
        P = model(images.float())
        loss=criterion(P, targets)
        loss.backward()
        opt.step()

        if batch_idx % args.log_interval == 0:
            val_loss, val_acc = evaluate('val', n_batches=4)
            train_loss = loss.data
            examples_this_epoch = batch_idx * len(images)
            epoch_progress = 100. * batch_idx / len(train_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
                epoch, examples_this_epoch, (data_length),
                epoch_progress, train_loss, val_loss, val_acc))

def evaluate(split, verbose=False, n_batches=None):
    '''
    Compute loss on val or test data.
    '''
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    if split == 'val':
        loader = val_loader
    elif split == 'test':
        loader = test_loader
    for batch_i, batch in enumerate(loader):
        data, target = batch['data'], batch['labels']
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data.float())
        loss += criterion(output, target, size_average=False).data
        # predict the argmax of the log-probabilities
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        n_examples += pred.size(0)
        if n_batches and (batch_i >= n_batches):
            break

    loss /= n_examples
    acc = 100. * correct / n_examples
    if verbose:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            split, loss, correct, n_examples, acc))
    return loss, acc

print('Begin training...')
# train the model one epoch at a time
for epoch in range(1, args.epochs + 1):
    train(epoch)
evaluate('test', verbose=True)

# Save the model (architecture and weights)
torch.save(model, args.model + '.pt')
# Later you can call torch.load(file) to re-load the trained model into python
# See http://pytorch.org/docs/master/notes/serialization.html for more details

