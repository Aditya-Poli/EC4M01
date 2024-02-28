import os
import argparse

import numpy as np
import torch

# import ddpg
import ddpg400 as ddpg
import utils

import env_phases as environment
# import environment
parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=2, help='num elements')
args = parser.parse_args()

env = environment.NOMA_IRS(num_elements=args.n, max_transmit_power=40)

file_name = f"{env.num_elements}"

if not os.path.exists(f"./Learning Curves/Sum Rate"):
    os.makedirs(f"./Learning Curves/Sum Rate")

if not os.path.exists("./Models"):
    os.makedirs("./Models")

#  seed
torch.manual_seed(env.seed)
np.random.seed(env.seed)

state_dim = env.state_dims
action_dim = env.action_dims
print(f"state_dims: {state_dim},   action_dims:{action_dim}")
max_action = 1

# device = torch.device(f"cuda:GPU" if torch.cuda.is_available() else "CPU")
device = "cpu"

actor_lr = 0.001
critic_lr = 0.001
actor_decay = 0.001
critic_decay = 0.001
discount = 0.99
tau = 0.001

ddpg_kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "actor_lr": actor_lr,
    "critic_lr": critic_lr,
    "actor_decay": actor_decay,
    "critic_decay": critic_decay,
    "device": device,
    "discount": discount,
    "tau": tau
    }

agent = ddpg.DDPG(**ddpg_kwargs)

buffer_size = 1000
batch_size = 500
replay_buffer = utils.ExperienceReplayBuffer(state_dim, action_dim,max_size=buffer_size)

rewards = []
max_reward = 0

num_eps = 20
num_time_steps_per_eps = 1000
env.max_time_steps = num_time_steps_per_eps

noise = utils.OrnsteinUlhenbeckActionNoise(mu=np.zeros(env.action_dims))

if not os.path.exists(f"./Models/{file_name}_best"):
    os.mkdir(f"./Models/{file_name}_best")

if not os.path.exists(f"./Learning Curves/Sum Rate/{file_name}"):
    os.mkdir(f"./Learning Curves/Sum Rate/{file_name}")

for episode_num in range(int(num_eps)):
    state, _, done = *env.reset(), False
    episode_reward = 0
    episode_time_steps = 0
    
    # state = whiten(state)

    eps_rewards = []

    for t in range(int(num_time_steps_per_eps)):
        # action from policy
        action = agent.predict(np.array(state)).flatten() + noise()
        action = np.clip(action,-1,1)
        action = action.flatten() + 1
        action /= 2
        # action = action + noise()

        next_state, reward, done, _, info = env.step(action)
        # t_reward = env.optimal
        t_reward = reward

        done = 1.0 if t == num_time_steps_per_eps - 1 else float(done)

        # store experience in replay buffer
        replay_buffer.add(state, action, next_state, t_reward, done)

        state = next_state
        episode_reward += t_reward

        # state = whiten(state)

        if t_reward > max_reward:
            max_reward = t_reward
            agent.save(f"./Models/{file_name}_best/{file_name}_best")

        # update parameters
        agent.update_parameters(replay_buffer, batch_size)

        # print(f"t: {t+1} ep: {episode_num + 1} Reward: {reward:.3f}")

        eps_rewards.append(t_reward)

        episode_time_steps += 1

    if done: 
        print(f"\nTotal T: {t+1} E: {episode_num + 1} E T: {episode_time_steps} Max. Reward: {max_reward:.3f} Avg. Reward: {np.average(eps_rewards)}\n")
        # reset the environment
        state, _, done = *env.reset(), False
        episode_reward = 0
        episode_time_steps = 0 
        episode_num += 1 

        noise.reset()

        # state = whiten(state)

        rewards.append(eps_rewards)

        np.save(f"./Learning Curves/Sum Rate/{file_name}/{file_name}_ep{episode_num + 1}", eps_rewards)
        

if not os.path.exists("./Env_Cache"):
    os.mkdir(f"./Env_Cache")
np.save(f"Env_Cache/optimal_{env.num_elements}",np.array(env.optimal_history))
np.save(f"Env_Cache/history_{env.num_elements}",np.array(env.history))
np.save(f"Env_Cache/opt_{env.num_elements}",np.array(env.opt_history))

if not os.path.exists(f"./Models/{file_name}"):
    os.mkdir(f"./Models/{file_name}")
agent.save(f"./Models/{file_name}/{file_name}")






