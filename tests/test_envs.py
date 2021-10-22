import random
import gym
import argparse
import envs

parser = argparse.ArgumentParser(description="Single Room Environment")

parser.add_argument(
    "--length",
    type=int,
    default=5,
    help="length of each wall of the length x length grid world",
)
parser.add_argument(
    "--num-tasks",
    type=int,
    default=4,
    help="length of each wall of the length x length grid world",
)

parser.add_argument(
    "--seeds",
    type=int,
    default=25,
    help="number of seeds to consider for averaging",
)
args = parser.parse_args()
"""Testing the environment"""
env = gym.make("NBottleneckClass-v0", args=args)
obs = env.reset()
"""Run for T timesteps and visualize"""

max_timesteps = 300
for _ in range(max_timesteps):
    action = random.randint(0, 3)
    next_obs, rew, done, task = env.step(action)
    obs = next_obs
