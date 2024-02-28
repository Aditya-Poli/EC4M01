import numpy as np
import matplotlib.pyplot as plt
import torch
import env_phases as environment
import utils
import ddpg
from scipy.io import loadmat

powers = [10, 30, 40, 50, 60, 80, 100, 120]
actor_lr = 0.001
critic_lr = 0.001
actor_decay = 0.001
critic_decay = 0.001
discount = 0.99
tau = 0.001
max_action = 1
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")


def power_rate(powers,n,path):
    envs = []
    sum_rates = []
    for p in powers:
        envs.append(environment.NOMA_IRS(num_elements=n,max_transmit_power=p))
    state_dim = envs[0].state_dims
    action_dim = envs[0].action_dims
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
    model = ddpg.DDPG(**ddpg_kwargs)
    model.load(f"./Models/{n}_best/{n}_best")
    for env in envs:
        sum_rate = []
        obs = env.reset()
        for i in range(env.max_time_steps):
            obs = env.state
            action= model.predict(obs)
            action = action.flatten()
            obs, reward, _, _, _ = env.step(action)
            sum_rate.append(env.optimal)
        sum_rate = max(sum_rate)
        sum_rates.append(sum_rate)
    return sum_rates


sum_rates_2 = power_rate(powers,2,"./Models/2_best/2_best")
sum_rates_4 = power_rate(powers,4,"./Models/4_best/4_best")
sum_rates_8 = power_rate(powers,8,"./Models/8_best/8_best")
sum_rates_16 = power_rate(powers,16,"./Models/16_best/16_best")

plt.figure(1)
plt.subplot(2,3,1)
plt.plot(powers,sum_rates_2,"-bo")
plt.plot(powers,sum_rates_4, "-r^")
plt.plot(powers,sum_rates_8, "-gs")
plt.plot(powers,sum_rates_16, "-yo")
plt.title("Power(${P}$) Vs Sum rate $R_{sum}$")
plt.xlabel(r"P $\rightarrow$")
plt.ylabel(r"$R_{sum} \rightarrow$")
# plt.semilogy()
plt.legend(["N=2","N=4","N=8","N=16"])
# plt.savefig("power_rsum_best.svg")
# plt.close()

plt.subplot(2,3,2)
elements = [2, 4, 8, 16]
rates = [sum_rates_2[2], sum_rates_4[2], sum_rates_8[2], sum_rates_16[2]]
plt.plot(elements, rates)
plt.xticks(elements,labels=elements)
# plt.savefig("N_rates.svg")
# plt.close()
mat = loadmat("Rsum_NOMA.mat")
print(mat["Rsum_sdr_avg"])
plt.plot(elements, mat["Rsum_sdr_avg"][0][1:-1])
plt.legend(["ddpg", "cvx"])



sum_rates_2 = power_rate(powers,2,"./Models/2/2")
sum_rates_4 = power_rate(powers,4,"./Models/4/4")
sum_rates_8 = power_rate(powers,8,"./Models/8/8")
sum_rates_16 = power_rate(powers,16,"./Models/16/16")

plt.subplot(2,3,3)
plt.plot(powers,sum_rates_2,"-bo")
plt.plot(powers,sum_rates_4, "-r^")
plt.plot(powers,sum_rates_8, "-gs")
plt.plot(powers,sum_rates_16, "-yo")
plt.title("Power(${P}$) Vs Sum rate $R_{sum}$")
plt.xlabel(r"P $\rightarrow$")
plt.ylabel(r"$R_{sum} \rightarrow$")
# plt.semilogy()
plt.legend(["N=2","N=4","N=8","N=16"])
# plt.savefig("power_rsum.svg")
# plt.show()




#-------------------------------------------------------------
#                           rewards
#-------------------------------------------------------------
rewardspath = "./Learning Curves/Sum Rate/"

def concat_rewards(n):
    path = rewardspath + str(n) + '/' + str(n) + '_ep'
    m = 2
    num_ep = 20
    ll = []
    llmax = []
    for i in range(num_ep):
        currpath = path + str(m) + '.npy'
        tmp = np.load(currpath)
        llmax.append(max(tmp))
        ll.append(tmp)
        m += 1
    return np.array(ll).flatten(), llmax

plt.subplot(2,3,4)
optimal_tmp = np.load("./Env_Cache/history_2.npy")
plt.plot(optimal_tmp)
optimal_tmp = np.load("./Env_Cache/history_4.npy")
plt.plot(optimal_tmp)
optimal_tmp = np.load("./Env_Cache/history_8.npy")
plt.plot(optimal_tmp)
optimal_tmp = np.load("./Env_Cache/history_16.npy")
plt.plot(optimal_tmp)
plt.legend(["2","4","8","16"])
# plt.savefig('rewards.svg')
plt.subplot(2,3,5)
optimal_tmp = np.load("./Env_Cache/optimal_2.npy")
# optimal_tmp = concat_rewards(16)
plt.plot(optimal_tmp)
optimal_tmp = np.load("./Env_Cache/optimal_4.npy")
# optimal_tmp = concat_rewards(16)
plt.plot(optimal_tmp)
optimal_tmp = np.load("./Env_Cache/optimal_8.npy")
# optimal_tmp = concat_rewards(16)
plt.plot(optimal_tmp)
optimal_tmp = np.load("./Env_Cache/optimal_16.npy")
# optimal_tmp = concat_rewards(16)
plt.plot(optimal_tmp)
plt.legend(["2", "4", "8", "16"])
# plt.savefig('optimal.svg')
plt.subplot(2,3,6)
plt.plot(concat_rewards(2)[1])
plt.plot(concat_rewards(4)[1])
plt.plot(concat_rewards(8)[1])
plt.plot(concat_rewards(16)[1])
# plt.savefig("max_optimal.svg")
plt.legend(["2", "4", "8", "16"])
plt.savefig("combined.svg")

plt.figure(2)
opt_8 = np.load("Env_Cache/opt_8.npy")
plt.plot(opt_8)
optimal_8 = np.load("Env_Cache/optimal_8.npy")
plt.plot(optimal_8)
print(f"opt: {np.average(opt_8)}, true: {np.average(optimal_8)}")
plt.legend(["opt", "true"])
plt.savefig("opt_true.svg")
