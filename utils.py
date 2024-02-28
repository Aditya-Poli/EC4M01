import numpy as np
import torch
from dataclasses import dataclass

def db2pow(pow):
    return 10**(pow/10)

def noise_power(noise_density,bandwidth):
    noiseVariancedBm = noise_density + 10*np.log10(bandwidth)
    return db2pow(noiseVariancedBm - 30)

def path_loss(exponent,distance):
    return -30 - exponent*10*np.log10(distance)

def db2powmw(pow):
  return db2pow(pow) / 10e3

def log10(arr):
    import cmath
    a = []
    for i in arr:
        a.append(cmath.log10(i))
    return np.array(a,dtype="complex64")

def sqrt(arr):
    import cmath
    a = []
    for i in arr:
        a.append(cmath.sqrt(i))
    return np.array(a,dtype="complex64")


class ExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.not_done[index]).to(self.device)
        )

class OrnsteinUlhenbeckActionNoise(object):
    def __init__(self, mu, sigma=0.2, theta=0.2, dt=5e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset() # reset the noise
        
    def __call__(self):
        x = self.x_prev + self.theta*(self.mu-self.x_prev)*self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
    
    def __repr__(self):
        return 'OrnsteinUlhenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def whiten(state):
    return (state - np.mean(state)) / np.std(state)

def moving_average(arr, smooth_factor = 0.5):
    x=smooth_factor  # smoothening factor
 
    i = 1
    # Initialize an empty list to
    # store exponential moving averages
    moving_averages = []
    
    # Insert first exponential average in the list
    moving_averages.append(arr[0])
    
    # Loop through the array elements
    while i < len(arr):
    
        # Calculate the exponential
        # average by using the formula
        window_average = round((x*arr[i])+
                            (1-x)*moving_averages[-1], 2)
        
        # Store the cumulative average
        # of current window in moving average list
        moving_averages.append(window_average)
        
        # Shift window to right by one position
        i += 1
    
    return moving_averages