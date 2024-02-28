import env_phases as env
import numpy as np
import matplotlib.pyplot as plt

env = env.NOMA_IRS()

done = env.done

while not done:
    state, reward, done, _, _ = env.step(env.action_space.sample())

opt_8 = np.load("Env_Cache/opt_8.npy")
plt.plot(opt_8)
optimal_8 = np.load("Env_Cache/optimal_8.npy")
plt.plot(optimal_8)
print(f"opt: {np.average(opt_8)}, true: {np.average(optimal_8)}")
plt.legend(["opt", "true"])
plt.savefig("opt_true.svg")

