
import gymnasium as gym
import gym_scheduler
import matplotlib.pyplot as plt

t0 = 1735689600
env = gym.make('gym_scheduler/Scheduler-v0', t0=t0, t1=t0+3600*24, render_mode='human')
observation, info = env.reset(seed=42)

terminated = False
while not terminated:
    action = env.action_space.sample()
    observation, reward, terminated, info = env.step(action)

