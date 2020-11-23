import minerl
import gym
import logging
import numpy as np
import torch

model = torch.load('./mineNet.pt')

def select_action(state, action, yaw, pitch):
    val = (model(state.long()).max(1).indices).long()

    binvec = dec2bin(val, 8).squeeze()

    action['forward'] = binvec[4]
    action['left'] = binvec[5]
    action['right'] = binvec[6]
    action['back'] = binvec[7]

    action['camera'] = [5*(binvec[0] - binvec[1]), 5*(binvec[2] - binvec[3])]
    return action

logging.basicConfig(level=logging.DEBUG)




env = gym.make('MineRLTreechop-v0')

obs  = env.reset()
done = False
net_reward = 0


num_episodes = 50

for i_ep in range(num_episodes):
    net_reward = 0
    yaw = 0
    pitch = 0
    
    done = False
    count = 0
    dur = 1000
    while not done:
 
        obs = torch.tensor(obs['pov'])
        obs = torch.transpose(obs, 0,2).unsqueeze(0).long()
        print(obs.shape)
        action = select_action(obs, env.action_space.noop(), yaw, pitch)
        yaw += 5*(binvec[0] - binvec[1])
        pitch += 5*(binvec[2] - binvec[3])

        obs, reward, done, info = env.step(action)
        net_reward += reward
        if count > dur:
            done = True
    obs = env.reset()
    print(net_reward)