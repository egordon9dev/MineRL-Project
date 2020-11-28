import gym
import minerl

env = gym.make("MineRLTreechop-v0")

obs = env.reset()
done = False

while not done:
    act = env.action_space.noop()
    act["camera"] = [1,0]
    act["attack"] = 1
    obs, rew, done, _ = env.step(act)