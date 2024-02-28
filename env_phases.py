import numpy as np
import random

from gymnasium import Env
from gymnasium import spaces

from utils import *

"""
Implementation of environment for the NOMA IRS
"""
class NOMA_IRS(Env):

    def __init__(self, num_users=3, num_elements=4, irs_distance=100,
                 bs_location=np.array([0,0]), irs_location=np.array([100,50]),
                 users_radius=10, max_transmit_power=50, bandwidth=1e6, noise_density=-174,
                 bs_irs_path_loss_exponent=2.4, irs_users_path_loss_exponent=2.8,
                 rician_factor=5, min_data_rate=0.05, noiseVar=0.01, seed=25,
                 max_time_steps=100):
        
        super(NOMA_IRS, self).__init__()

        np.random.seed(seed=seed)
        random.seed(seed)
        self.seed = seed

        self.num_users = num_users
        self.num_elements = num_elements
        self.irs_distance = irs_distance
        self.bs_location = bs_location
        self.irs_location = irs_location
        self.users_radius = users_radius
        self.max_transmit_power = max_transmit_power
        self.bandwidth = bandwidth
        self.noise_density = noise_density
        self.bs_irs_path_loss_exponent = bs_irs_path_loss_exponent
        self.irs_users_path_loss_exponent = irs_users_path_loss_exponent
        self.rician_factor = rician_factor
        self.min_data_rate = min_data_rate
        self.noiseVar = noiseVar
        self.max_time_steps = max_time_steps

        # time step counter
        self.time_step_cntr = 0

        # generate user positions and calculate the distances
        self.user_positions = self.generate_user_positions()

        # calculate distances
        self.distance_bs_irs = np.linalg.norm(self.irs_location - self.bs_location)
        self.distance_irs_users = np.linalg.norm(self.user_positions - self.irs_location, axis=1)

        # calculate the noise power (sigma sqr) and its inv
        self.sigma_sqr = noise_power(self.noise_density, self.bandwidth)
        self.sigma_sqr_inv = self.sigma_sqr ** -1

        # calculate the path loss
        # i) b/w BS and IRS
        # ii) b/w IRS and UE's
        self.beta_BS_IRS = path_loss(self.bs_irs_path_loss_exponent,self.distance_bs_irs)
        self.beta_IRS_UE = path_loss(self.irs_users_path_loss_exponent,self.distance_irs_users)

        # channel between IRS and UE (Rayleigh Fading)
        self.hk = self.channel_IRS_UE()
        # channel between BS and IRS (Rician Fading)
        self.h_BS = self.channel_BS_IRS()

        # define observation space and action space
        self.action_dims = self.num_elements # N
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dims,), dtype="float32")
        self.state_dims = 2*self.num_users + self.action_dims # 2K + N
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dims,), dtype="float32")

        # max power coefficents
        self.optimal_coefficients = np.ones((self.num_users,))

        # intial state
        self.pk = None; self.phi = None; self.pk_r = None
        self.state = self.init_state()

        self.history = []
        self.optimal_history = []
        self.opt_history = []
        self.prev_action = self.phi.flatten()
        self.prev_rate = 0
        self.optimal = None
        self.opt = None
        self.done = self.time_step_cntr == self.max_time_steps
        self.info = {"Rsum" : 0}
    
    def init_state(self):
        # the below 2 lines are commented because for every max time steps the
        # rewards history is periodic
        self.pk = np.ones((self.num_users,))
        self.phi = np.zeros((self.num_elements,))
        self.update_powers(self.phi.flatten())
        return np.hstack((self.pk.flatten(),self.pk_r.flatten(),self.phi.flatten())).astype("float32")
    
    def update_powers(self,action):
        p_tx = self.pk * self.max_transmit_power
        self.phi = action[-self.num_elements:]
        p_rx = np.abs(self.h_BS.conj().T @ np.diag(np.exp(self.phi*2j*np.pi)) @ self.hk)**2 * p_tx
        self.pk_r = p_rx / self.max_transmit_power

    def generate_user_positions(self):
        # Generate user positions uniformly distributed on a circle
        angles = np.linspace(0, 2 * np.pi, self.num_users, endpoint=False)
        user_positions = np.array([
            self.users_radius * np.cos(angles),
            self.users_radius * np.sin(angles)
        ]).T + np.array([200, 0])
        return user_positions
    
    def channel_IRS_UE(self):
        '''Rayleigh Fading'''
        shape = (self.num_elements,self.num_users)
        hk = (1/np.sqrt(2)) * (np.random.randn(*shape)+1j*np.random.randn(*shape))
        hk = sqrt(db2pow(self.beta_IRS_UE)) * hk
        return hk.astype("complex64").reshape((self.num_elements,self.num_users))
    
    def channel_BS_IRS(self):
        '''Rician Fading'''
        antennaSpacing = 0.5 # half wavelength lambda/2
        pos_BS = complex(*self.bs_location)
        pos_IRS = complex(*self.irs_location)
        angleBS_IRS = np.angle(pos_IRS - pos_BS)
        h_LOS = np.exp(1j*2*np.pi*np.arange(self.num_elements)*np.sin(angleBS_IRS)*antennaSpacing)
        '''NEEED TO CHECK'''
        h_BS = np.sqrt(self.rician_factor/(self.rician_factor+1))*h_LOS + \
        np.sqrt(1/(self.rician_factor+1))*(1/np.sqrt(2))*(np.random.randn(self.num_elements,) + \
        1j*np.random.randn(self.num_elements,))
        h_BS = np.sqrt(db2pow(self.beta_BS_IRS))*h_BS
        return h_BS.reshape((self.num_elements),1)
    
    def update_users(self):
        self.user_positions = self.generate_user_positions()
        self.distance_irs_users = np.linalg.norm(self.user_positions - self.irs_location, axis=1)
        self.beta_IRS_UE = path_loss(self.irs_users_path_loss_exponent,self.distance_irs_users)
    
    def update_channels(self):
        self.hk = self.channel_IRS_UE()
        # self.h_BS = self.channel_BS_IRS()
    
    def step(self, action):
        self.time_step_cntr += 1
        self.update_powers(action=action)
        
        reward, original, opt = self.reward()
        self.prev_rate = original
        self.optimal = original
        self.opt = opt
        self.optimal_history.append(original)
        self.history.append(reward)
        self.opt_history.append(opt)
        self.info["Rsum"] = reward

        self.state = self.get_state().astype("float32")
        self.prev_action = action
        self.done = (self.time_step_cntr == self.max_time_steps) or (self.prev_rate >= self.opt)

        truncation = False
        self.update_users()
        self.update_channels()
        # next_state, reward, done, truncation, info
        return self.state, reward, self.done, truncation, self.info
    
    def get_state(self):
        return np.hstack((self.pk.flatten(), self.pk_r.flatten(), self.prev_action.flatten()))
    
    def reward(self):
        phases = self.phi
        t_reward, opt = self.sum_rate(phases)
        reward_t = t_reward
        if(reward_t < self.prev_rate):
            reward_t -= self.prev_rate
        return reward_t, t_reward, opt
    
    # def sum_rate(self,phases):
    #     g_phi = self.h_BS.conj().T @ np.diag(np.exp(2j*phases*np.pi))

    #     y = 0
    #     for i, pk in enumerate(self.pk * self.max_transmit_power):
    #         y += ((np.abs(g_phi @ self.hk[:,i])**2) * pk)
    #     sum_rate = np.log2(1 + (self.sigma_sqr_inv * y))[0]
    #     return sum_rate
    
    def sum_rate(self,phases):
        optimal = np.sum([self.rate(pk_r) for pk_r in self.pk_r*self.max_transmit_power])
        sum_rate = self.rate(np.sum(self.pk_r * self.max_transmit_power))
        return sum_rate, optimal
    
    def rate(self,pk_r):
        return np.log2(1 + (self.sigma_sqr_inv * pk_r))
    
    
    def reset(self, seed=25):
        self.time_step_cntr = 0
        self.state = self.init_state()
        self.update_channels()
        self.update_users()
        self.info["Rsum"] = 0
        return self.state, self.info
    
    def render(self, mode="human"):
        pass

