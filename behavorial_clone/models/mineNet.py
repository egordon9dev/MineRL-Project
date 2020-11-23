import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################

        self.conv_layer = nn.Conv2d(im_size[0], hidden_dim, kernel_size)
        
        height = im_size[1]
        width = im_size[2]
        
        pool_size = 3
        self.pooling_layer = nn.MaxPool2d(pool_size, stride=1)
        
        self.out_with_pool =  nn.Linear(hidden_dim * (height - kernel_size + 2 - pool_size) * (width - kernel_size + 2 - pool_size), n_classes)
        self.out = nn.Linear(hidden_dim * (height - kernel_size + 1) * (width - kernel_size + 1), n_classes)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):

        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        
        h = self.conv_layer(images)
        h = F.relu(h)
        h = self.pooling_layer(h)
        
        N = images.size(0)
        h = h.view(N, -1)
        
        scores = (self.out_with_pool(h))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

