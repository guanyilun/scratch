import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import gym_scheduler
import matplotlib.pyplot as plt

t0 = 1735689600
env = gym.make('gym_scheduler/Scheduler-v0', t0=t0, t1=t0+3600*24*5, render_mode='human')
env = FlattenObservation(env)
observation, info = env.reset(seed=42)
terminated = False
while not terminated:
    action = env.action_space.sample()
    observation, reward, terminated, done, info = env.step(action)
    print("reward:", float(reward))
