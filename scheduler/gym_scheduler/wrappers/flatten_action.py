import gym


class FlattenAction(gym.Wrapper):
    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(env.action_space)

    def step(self, action):
        # Flatten the action
        unflatten_action = gym.spaces.utils.unflatten(self.env.action_space, action)
        on = 1 if action[4] > 0.5 else 0  # need to update when action sequence order changes
        unflatten_action['on'] = on
        # Call the step method of the wrapped environment with the flattened action
        obs, reward, terminated, done, info = self.env.step(unflatten_action)
        # Return the flattened observation and other outputs
        return obs, reward, terminated, done, info

    def reset(self, seed=None):
        # Call the reset method of the wrapped environment
        obs = self.env.reset()
        # Return the flattened observation
        return obs
