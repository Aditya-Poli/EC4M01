import os

import numpy as np
import utils

import environment
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

log_path = os.path.join('Training', 'Logs')

env = environment.NOMA_IRS(num_elements=2,max_time_steps=1000)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, batch_size=1000, \
             buffer_size=2_000, seed=25, tensorboard_log=log_path)
model.learn(total_timesteps=20_000,log_interval=1)

model.save("./Training/Models/n{env.num_elements}_{model.total_timesteps}_{model.buffer_size}")
