import numpy as np
import random

from gymnasium import Env
from gymnasium import spaces

from utils import *


class NOMA_IRS(Env):
    def __init__(self, num_users=3, num_elements=4, irs_distance=100,
                 bs_location=np.array([0, 0]), irs_location=np.array([100, 50]),
                 users_radius=10, max_transmit_power=50, bandwidth=1e6, noise_density=-174,
                 bs_irs_path_loss_exponent=2.4, irs_users_path_loss_exponent=2.8,
                 rician_factor=5, min_data_rate=0.05, noiseVar=0.01, seed=25, max_episodes=100):

        np.random.seed(seed=seed)
        random.seed(seed)
        self.seed = seed

        super(NOMA_IRS, self).__init__()

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
        self.max_episodes = max_episodes

        self.episode_cntr = 0

        # generate user positions and calculate the distances
        self.user_positions = self.generate_user_positions()

        self.distance_bs_irs = np.linalg.norm(self.irs_location - self.bs_location)
        self.distance_irs_users = np.linalg.norm(self.user_positions - self.irs_location, axis=1)

        # calaculate the noise power (sigma_sqr)
        self.sigma_sqr = noise_power(self.noise_density,self.bandwidth)

        # calculate the path loss between BS and IRS
        # & IRS and UE's
        self.beta_BS_IRS = -30 -self.bs_irs_path_loss_exponent*10*np.log10(self.distance_bs_irs)
        self.beta_IRS_UE = -30 -self.irs_users_path_loss_exponent*10*np.log10(self.distance_irs_users)

        # channel between IRS and UE (RayleighFading)
        self.hk = self.channel_IRS_UE()

        # channel between BS and IRS (Rician Fading)
        self.h_BS = self.channel_BS_IRS()


        # Define observation space and action space
        # number of users -> IRS and irs -> BS
        # powers of users and Re(phi) + Im(phi) for num_elements
        self.action_dims = self.num_elements # N 
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dims,),dtype="float32")
        # state will be Tx and Rx powers for each user
        # and also the previous action
        self.state_dims = 2*self.num_users + self.action_dims # 2K + (N)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dims,),dtype="float32")

        # optimal coeff's based on distance
        # self.optimal_coefficients = self.distance_irs_users / np.sum(self.distance_irs_users)
        self.optimal_coefficients = np.ones((self.num_users,))
        # initial state
        self.pk = None
        self.phi = None
        self.pk_r = None
        self.state = self.init_state()


        self.prev_action = self.phi
        self.info = {"Rsum":0}

        self.done = self.episode_cntr == self.max_episodes

        self.noiseVar = noiseVar
        # vComplexNoise = sqrt(noiseVar / 2) * (randn(1, numSamples) + (1i * randn(1, numSamples)))
        self.CNoise = lambda nSamples: np.sqrt(self.noiseVar / 2) * \
         (np.random.randn(1,nSamples) + 1j*np.random.randn(1,nSamples))

        self.history = []
        self.optimal = None
        self.prev_rate = 0
        self.optimal_history = []



    def init_state(self):
        self.pk = np.zeros((self.num_users,))
        self.phi = np.zeros((self.num_elements,))
        self.update_powers(self.phi.flatten())
        # received power at BS for every user
        # p_rx = np.ones((self.num_users,))*0.25
        return np.hstack((self.pk.flatten(),self.pk_r.flatten(),self.phi.flatten())).astype("float32")

    def update_powers(self,action):
        p_tx = self.optimal_coefficients*self.max_transmit_power
        self.phi = action
        p_rx = (np.abs(self.h_BS.conj().T @ np.diag(self.phi*2j*np.pi) @ self.hk)**2 * np.sqrt(p_tx))
        self.pk = p_tx
        self.pk_r = p_rx


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
        # self.optimal_coefficients = self.distance_irs_users / np.sum(self.distance_irs_users)
        self.beta_IRS_UE = -30-28*np.log10(self.distance_irs_users)

    def update_channels(self):
        self.hk = self.channel_IRS_UE()
        # self.h_BS = self.channel_BS_IRS() # not updated


    def step(self, action):
        # state, reward, done, info
        self.episode_cntr += 1
        # apply action
        self.update_powers(action)
        reward,original = self.reward(action)
        self.prev_rate = original
        self.optimal = original
        self.optimal_history.append(original)
        # # update the user locations
        # self.update_users()
        # # update the channels
        # self.update_channels()
        #get state
        self.state = self.get_state().astype("float32")
        self.prev_action = action
        self.done = self.episode_cntr == self.max_episodes
        # if self.done:
        #     self.reset()
        #     reward = 0
        truncation = False
        self.info["Rsum"] =  reward
        self.history.append(reward)
        '''In SB3 truncate needs to be returned'''
        # print({"state":self.state,"dtype":self.state.dtype,"shape":self.state.shape})
        return self.state,reward,self.done,truncation,self.info

    def get_state(self):
        return np.hstack((self.pk.flatten(),self.pk_r.flatten(),self.prev_action.flatten()))

    def reward(self,action):
        phases = action[:self.num_elements]
        t_reward = self.rate(self.pk,phases)
        # optimal power allocation
        # p_reward = [0 if p <= q else -p*self.max_transmit_power for p,q in zip(action[self.num_elements:],self.optimal_coefficients)]
        # p_reward = sum(p_reward)
        reward = t_reward
        if(t_reward <= self.prev_rate):
          reward = -(self.prev_rate - t_reward)
        return reward, t_reward

    def rate(self,powers,phases):
        g_phi = self.h_BS.conj().T @ np.diag(np.exp(2j*phases*np.pi))
        sigma2_inv = self.sigma_sqr ** -1
        y = 0
        for i,pk in enumerate(db2powmw(powers*self.max_transmit_power)):
            y += np.abs(g_phi @ self.hk[:,i])**2 * pk
        sum_rate = np.log2(sigma2_inv * y)
        return sum_rate[-1]

    @staticmethod
    def sumRate(summ):
        return np.log2(1 + (summ * 4 * np.pi * np.pi))

    def reset(self,seed=26):
        self.episode_cntr = 0
        # self.done = False
        # Reset the environment to an initial state
        self.state = self.init_state()
        # update the user locations
        self.update_users()
        # update the channels
        self.update_channels()
        '''In SB3 seed need to passed to reset and it should return (obs,info)'''
        self.info["Rsum"] = 0
        return self.state, self.info

    def render(self, mode='human'):
        # Implement a rendering method for visualization
        pass
