import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(Actor, self).__init__()
        
        hidden_dim = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()
        self.device = device

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.max_action = 1

    def forward(self, state):
        a = torch.tanh(self.l1(state.float()))

        # bactch normalisation
        a = self.bn1(a)
        a = torch.tanh(self.l2(a))

        a = self.bn2(a)
        # a = torch.tanh(self.l3(a))
        a = torch.tanh(self.l3(a))

        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        hidden_dim = 1 if (state_dim + action_dim) == 0 else 2 ** ((state_dim + action_dim) - 1).bit_length()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, state, action):
        q = torch.tanh(self.l1(state.float()))

        q = self.bn1(q)
        q = torch.tanh(self.l2(torch.cat([q, action], 1)))

        q = self.l3(q)

        return q


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, actor_lr, critic_lr, actor_decay, critic_decay, device, discount=0.99, tau=0.001):
        self.device = device

        # Actor network and optimizer
        self.actor = Actor(state_dim, action_dim,device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=critic_lr, weight_decay=critic_decay)

        # critic network
        self.critic = Critic(state_dim, action_dim).to((self.device))
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=critic_decay)

        # discount and polyak coefficient
        self.discount = discount
        self.tau = tau


    def predict(self, state):
        self.actor.eval()

        state = torch.FloatTensor(state.reshape(1,-1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten().reshape(1, -1)

        return action

    def update_parameters(self, replay_buffer, batch_size=16):
        self.actor.train()

        # sample from experience replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        # error of devices
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        not_done = not_done.to(self.device)

        # compute the target Q-Value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # compute current Q-Value estimate
        current_Q = self.critic(state, action)

        # compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # Save the model parameters
    def save(self, file_name):
        torch.save(self.critic.state_dict(), file_name + "_critic")
        torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")

        torch.save(self.actor.state_dict(), file_name + "_actor")
        torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")

    # Load the model parameters
    def load(self, file_name):
        self.critic.load_state_dict(torch.load(file_name + "_critic",map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(file_name + "_critic_optimizer",map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(file_name + "_actor",map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(file_name + "_actor_optimizer",map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)



        
