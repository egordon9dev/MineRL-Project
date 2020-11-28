from network import ReplayMemory, DQN, Transition
from environment import MyEnv
import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os

env = MyEnv()
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50000
TARGET_UPDATE = 10
LR = 0.005
test_time = True

n_steps = 8
n_actions = env.action_space.n
img_height = 64
img_width = 64
policy_net = None
network_path = "target_net.pt"
if os.path.exists(network_path):
    policy_net = torch.load(network_path)
    print("successfully loaded existing network from file: " + network_path)
else:
    policy_net = DQN(img_height, img_width, n_actions)
target_net = DQN(img_height, img_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)

steps_done = 0
logfile = "train_log.txt"
with open(logfile, "w+") as f:
    f.write("CS4803 MineRL Project Logs:\n")
def append_log(s):
    with open(logfile, "a") as f:
        f.write(s + "\n")

def state_from_obs(obs):
    # get the camera image from the observation dict and convert the image
    # to the correct shape: (C, H, W)
    img = torch.tensor(obs["pov"] / 255.0, dtype=torch.float32)
    flat_img = torch.transpose(torch.transpose(img, 0, 1), 0, 2).reshape(64*64*3)
    compass_angle = torch.tensor(obs["compassAngle"] / 180.0, dtype=torch.float32).reshape(1)
    return torch.cat([flat_img, compass_angle])

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold or test_time:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(torch.unsqueeze(state, 0)).max(1)[1].view(1, 1)
    else:
        explore_act = random.randrange(n_actions)
        return torch.tensor([[explore_act]], device=device, dtype=torch.long)

ep_durations = []
ep_losses = []
ep_reward = []

def update_plots():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(ep_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())
    plt.savefig("durations_plot.png")

    plt.figure(2)
    plt.clf()
    losses_t = torch.tensor(ep_losses, dtype=torch.float)
    plt.title("Training Loss")
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(losses_t.numpy())
    if len(losses_t) >= 10:
        means = losses_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())
    plt.savefig("losses_plot.png")

    plt.figure(2)
    plt.clf()
    reward_t = torch.tensor(ep_reward, dtype=torch.float)
    plt.title("Training Reward")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_t.numpy())
    if len(reward_t) >= 10:
        means = reward_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())
    plt.savefig("rewards_plot.png")

losses = []
def optimize_model():
    global losses
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    state_batches = [torch.cat([torch.unsqueeze(s,0) for s in batch.states[i]]) for i in range(n_steps)]
    action_batches = [torch.cat(batch.actions[i]) for i in range(n_steps)]
    reward_batches = [torch.cat(batch.rewards[i]) for i in range(n_steps)]

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = [policy_net(state_batches[i]).gather(1, action_batches[i]) for i in range(n_steps)]

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = [torch.zeros(BATCH_SIZE, device=device) for i in range(n_steps)]
    for i in range(n_steps):
        non_final_next_states = torch.cat([torch.unsqueeze(s, 0) for s in batch.next_state
                                                if s is not None])
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        next_state_values[i][non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    losses.append(loss.detach().numpy())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 500
print("beginning training...")
for i_episode in range(num_episodes):
    # Initialize the environment and state
    obs0 = env.reset()
    states = [None] * n_steps
    state = state_from_obs(obs0)
    ep_rew = 0
    for t in count():
        # Select and perform an action
        action = select_action(state)
        obs, rew, done, _ = env.step(action)
        ep_rew += rew
        reward = torch.tensor([rew], device=device)
        next_state = state_from_obs(obs)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if t == 999 or done:
            ep_losses.append(np.mean(losses))
            losses = []
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            logstr = f"Episodes: {i_episode+1}\tepsilon: {epsilon}\tduration: {t+1}\tloss: {ep_losses[-1]}\treward: {ep_rew}"
            ep_reward.append(ep_rew)
            print(logstr)
            append_log(logstr)
            update_plots()
            torch.save(target_net, "target_net.pt")
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.close()
plt.ioff()
plt.show()
