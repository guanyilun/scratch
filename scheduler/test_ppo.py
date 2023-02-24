import gym

import gym_scheduler
from gym_scheduler.wrappers.flatten_action import FlattenAction

from stable_baselines3 import PPO

t0 = 1735689600
env = gym.make('gym_scheduler/Scheduler-v0', t0=t0, t1=t0+3600*24*5, render_mode='human')
env = FlattenAction(env)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
model.save("model")

