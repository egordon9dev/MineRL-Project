import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvQNet(nn.Module):
    def __init__(self, env, config, logger=None):
        super().__init__()

        #####################################################################
        # TODO: Define a CNN for the forward pass.
        #   Use the CNN architecture described in the following DeepMind
        #   paper by Mnih et. al.:
        #       https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        #
        # Some useful information:
        #     observation shape: env.observation_space.shape -> (H, W, C)
        #     number of actions: env.action_space.n
        #     number of stacked observations in state: config.state_history
        #####################################################################
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4, padding=0)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0)

        H,W,C = env.observation_space.shape


        lin_size = ((((W - (8-1) - 1) // 4 + 1) - (4 - 1) - 1) // 2 + 1) \
        * ((((H - (8-1) - 1) // 4 + 1) - (4 - 1) - 1) // 2 + 1) * 32
        
        self.final1 = nn.Linear(lin_size, 256)
        self.final3 = nn.ReLU()
        self.final = nn.Linear(256, env.action_space.n)

        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################

    def forward(self, state):
        #####################################################################
        # TODO: Implement the forward pass.
        #####################################################################
        penul = F.relu(self.conv2(F.relu(self.conv1(state.permute([0,3,1,2])))))
        actions = self.final(self.final3(self.final1((penul.view(penul.size(0), -1)))))
        return actions
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
