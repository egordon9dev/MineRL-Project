import minerl
import gym
env = gym.make('MineRLTreechop-v0')


obs  = env.reset()
done = False
net_reward = 0

while not done:
    action = env.action_space.noop()

    action['camera'] = [0, 0.03]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 1
    action['attack'] = 1

    obs, reward, done, info = env.step(
        env.action_space.sample())

    net_reward += reward
    print("Total reward: ", net_reward)