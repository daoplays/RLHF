from distutils.util import strtobool
import gymnasium as gym
import gym_moving_dot

import numpy as np
import torch
from train import Agent

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode="human")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


envs = gym.vector.SyncVectorEnv(
    [make_env("MovingDotDiscreteNoFrameskip-v0", 0, i, False, "test") for i in range(1)]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = Agent(envs).to(device)


agent.load_state_dict(torch.load("./rlhf_trained"))
agent.eval()

next_obs = torch.Tensor(envs.reset()[0]).to(device)

while(True):
    action, logprob, value = agent.get_action_and_value(next_obs)
    next_obs, reward, truncated, terminated, info = envs.step(action.cpu().numpy())
    done = np.logical_or(truncated,terminated)
    next_obs = torch.Tensor(next_obs).to(device)

    if (done):
        break

envs.close()