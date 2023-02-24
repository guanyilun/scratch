from gym.envs.registration import register

register(
    id="gym_scheduler/Scheduler-v0",
    entry_point="gym_scheduler.envs:SchedulerEnv",
)
