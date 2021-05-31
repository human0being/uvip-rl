import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import trange
import torch
from torch.distributions import Categorical
import torch.nn as nn
import CartPoleEnv
from sklearn.neighbors import NearestNeighbors
import importlib
import utils
import scipy
from numba import njit, prange
import matplotlib as mpl
import random
from tqdm import tqdm_notebook

importlib.reload(utils)
importlib.reload(CartPoleEnv)

def generate_samples_given(env, X_samples, N=1):
    device = 'cpu'
    rewards0 = []
    rewards1 = []
    for X in X_samples:
        env.reset(X)
        _, r0, _, _ = env.step(0)
        env.reset(X)
        _, r1, _, _ = env.step(1)
        rewards0.append(r0)
        rewards1.append(r1)

    rewards0 = torch.tensor(rewards0, device=device, dtype=torch.float32)
    rewards1 = torch.tensor(rewards1, device=device, dtype=torch.float32)
    rewards = torch.cat((rewards0[None, :], rewards1[None, :]), axis=0)

    return torch.FloatTensor(X_samples), rewards

def generate_samples_uniform(env, N=1):
    device = 'cpu'
    X0_samples = torch.FloatTensor(N, 1).uniform_(-2.3, 2.3)
    X1_samples = torch.FloatTensor(N, 1).uniform_(-2., 2.)
    X2_samples = torch.FloatTensor(N, 1).uniform_(-0.2, 0.2)
    X3_samples = torch.FloatTensor(N, 1).uniform_(-1.1, 1.1)
    X_samples = torch.cat((X0_samples, X1_samples, X2_samples, X3_samples), dim=-1).to(device)

    rewards0 = []
    rewards1 = []
    for X in X_samples:
        env.reset(X.numpy())
        _, r0, _, _ = env.step(0)
        env.reset(X.numpy())
        _, r1, _, _ = env.step(1)
        rewards0.append(r0)
        rewards1.append(r1)

    rewards0 = torch.tensor(rewards0, device=device, dtype=torch.float32)
    rewards1 = torch.tensor(rewards1, device=device, dtype=torch.float32)
    rewards = torch.cat((rewards0[None, :], rewards1[None, :]), axis=0)

    return X_samples, rewards

def generate_samples_normal(env, N=1):
    device = 'cpu'
    X0_samples = torch.FloatTensor(N, 1).normal_(0, 2.0 / 3.)
    X1_samples = torch.FloatTensor(N, 1).normal_(0, 2.0 / 3.)
    X2_samples = torch.FloatTensor(N, 1).normal_(0, 0.25 / 3.)
    X3_samples = torch.FloatTensor(N, 1).normal_(0, 1.1 / 3.)
    X_samples = torch.cat((X0_samples, X1_samples, X2_samples, X3_samples), dim=-1).to(device)

    rewards0 = []
    rewards1 = []
    for X in X_samples:
        env.reset(X.numpy())
        _, r0, _, _ = env.step(0)
        env.reset(X.numpy())
        _, r1, _, _ = env.step(1)
        rewards0.append(r0)
        rewards1.append(r1)

    rewards0 = torch.tensor(rewards0, device=device, dtype=torch.float32)
    rewards1 = torch.tensor(rewards1, device=device, dtype=torch.float32)
    rewards = torch.cat((rewards0[None, :], rewards1[None, :]), axis=0)

    return X_samples, rewards

def predict_probs(states, policy):
    with torch.no_grad():
        states = torch.as_tensor(states).to(torch.float32)
        probs = torch.softmax(policy(states), dim=-1)
        return probs.numpy()

def select_action(model, state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    m = Categorical(probs)

    action = m.sample()

    return action.item()

def VpiEstimation(env, X_samples, policy, alg_type, n_samples=15, gamma=0.99):
    # n_samples = 15
    n_actions = 2
    Vpi = []

    step = 0
    for step in tqdm_notebook(range(X_samples.shape[0])):
        average_reward = 0.
        for k in range(n_samples):
            discounted_reward = 0.
            s = env.reset(X_samples[step].numpy())
            cur_gamma = 1.
            while True:
                if alg_type == 'a2c':
                    a = select_action(policy, np.array([s]))
                elif alg_type == 'LD':
                    a = env.getBestAction(s)
                elif alg_type == 'random':
                    a = np.random.choice(range(2))

                new_s, r, done, _ = env.step(a)
                discounted_reward += cur_gamma * r
                s = new_s
                cur_gamma *= gamma
                if done:
                    break

            average_reward += discounted_reward

        Vpi.append(average_reward / n_samples)

    return torch.tensor(Vpi, dtype=torch.float32)


@njit(nopython=True, parallel=True)
def calc(Txa, X_samples_pi, cov):
    probs0 = np.empty((0, X_samples_pi.shape[0]), dtype=np.float32)
    probs1 = np.empty((0, X_samples_pi.shape[0]), dtype=np.float32)
    inv = np.linalg.inv(cov)
    for i in prange(Txa.shape[1]):
        d1 = X_samples_pi - Txa[0, i, :]
        d2 = X_samples_pi - Txa[1, i, :]
        s1 = (d1 @ inv) @ d1.T
        s2 = (d2 @ inv) @ d2.T
        p1 = np.exp(-s1 / 2.) / np.sqrt(2 * (np.pi**4) * np.linalg.det(cov))
        p2 = np.exp(-s2 / 2.) / np.sqrt(2 * (np.pi**4) * np.linalg.det(cov))
        probs0 = np.append(probs0, p1.astype(np.float32), axis=0)
        probs1 = np.append(probs1, p2.astype(np.float32), axis=0)
        if i % 1000 == 0:
            print(i)

    return probs0, probs1


def getMonteCarloUpperBounds(env, policy, X_samples, rewards, V_pi, k=4, total_steps=50, M1=150, M2=150, gamma=0.99):
    """
    0,1 means action 
    """
    max_grad_norm = 5000
    loss_history = []
    grad_norm_history = []
    eval_freq = 1
    device = 'cpu'

    state_dim, n_actions = env.observation_space.shape, env.action_space.n
    # X_samples, rewards = generate_samples_uniform(env, 2000)
    N = X_samples.shape[0]
    neigh = NearestNeighbors(n_neighbors=5, algorithm='kd_tree', leaf_size=30, n_jobs=-1)
    neigh.fit(X_samples.tolist())
    rewards = rewards.reshape(n_actions, N, 1)
    # V_pi = VpiEstimation(X_samples, policy)
    V_up = torch.clone(V_pi) + 5
    perm = torch.randperm(X_samples.size(0))
    idx = perm[:50] #for output
    # idx = range(N)
    N_states = 50
    samples = X_samples[idx] #for output
    single_sample = np.random.randint(0, X_samples.shape[0], size=1)
    upper_bound_sample = []
    # print(X_samples[0])
    # assert False
    upper_list = [V_up.numpy()]
    norm_list_upper = []
    step = 0
    YX0 = []
    YX1 = []
    for i in range(N):
        YX0j = []
        YX1j = []
        for j in range(M1):
            env.reset(X_samples[i].numpy())
            YXi0, _, _, _ = env.step(0)
            env.reset(X_samples[i].numpy())
            YXi1, _, _, _ = env.step(1)
            YX0j.append(YXi0)
            YX1j.append(YXi1)

        YX0.append(YX0j)
        YX1.append(YX1j)

    YX0 = torch.tensor(YX0, device=device, dtype=torch.float32)
    YX1 = torch.tensor(YX1, device=device, dtype=torch.float32)
    Yxa_M1 = torch.cat((YX0[None, :, :, :], YX1[None, :, :, :]), axis=0)
    _, idxes_neigh1 = neigh.kneighbors(torch.flatten(Yxa_M1, end_dim=2).numpy().tolist())
    V_pi1 = V_pi[torch.tensor(idxes_neigh1)].mean(dim=-1).reshape(n_actions, N, M1)
    V_mean = torch.mean(V_pi1, dim=-1) #same shape as rewards
    V_mean = V_mean.reshape(n_actions, N, 1)
    with trange(step, total_steps + 1) as progress_bar:
        for step in progress_bar:
            YX0 = []
            YX1 = []
            for i in range(N):
                YX0j = []
                YX1j = []
                for j in range(M2):
                    env.reset(X_samples[i].numpy())
                    YXi0, _, _, _ = env.step(0)
                    env.reset(X_samples[i].numpy())
                    YXi1, _, _, _ = env.step(1)
                    YX0j.append(YXi0)
                    YX1j.append(YXi1)

                YX0.append(YX0j)
                YX1.append(YX1j)

            YX0 = torch.tensor(YX0, device=device, dtype=torch.float32)
            YX1 = torch.tensor(YX1, device=device, dtype=torch.float32)
            Yxa_M2 = torch.cat((YX0[None, :, :, :], YX1[None, :, :, :]), axis=0)
            #Yxa has shape (2, N, M1 + M2, 4)
            _, idxes_neigh2 = neigh.kneighbors(torch.flatten(Yxa_M2, end_dim=2).numpy().tolist())
            V_pi2 = V_pi[torch.tensor(idxes_neigh2)].mean(dim=-1).reshape(n_actions, N, M2)
            V_k = V_up[torch.tensor(idxes_neigh2)].mean(dim=-1).reshape(n_actions, N, M2)

            M = V_pi2 - V_mean
            # EM = torch.mean(M, dim=-1)
            # DM = torch.sum(M**2, dim=-1) / (M.shape[-1] - 1)
            #M has shape (n_actions, N, M2)
            V_up = (rewards + gamma * (V_k - M)).max(dim=0)[0].mean(dim=-1)

            if step % eval_freq == 0:
                clear_output(True)
                plt.figure(figsize=(15, 10))
                plt.subplot(121)
                lower = V_pi[idx].numpy()
                upper = V_up[idx].numpy()
                upper_list.append(upper)
                upper_bound_sample.append(V_up[single_sample])
                upper_plot = np.repeat(upper.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
                lower_plot = np.repeat(lower.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
                states_plot = np.concatenate((np.arange(N_states).reshape(-1,1),
                                          (np.arange(N_states)+1).reshape(-1,1)), axis=1).reshape(-1)
                plt.fill_between(states_plot, lower_plot, upper_plot, alpha=0.3,
                            edgecolor='k', linestyle='-')
                plt.plot(states_plot, upper_plot, 'b')
                plt.legend(loc="upper left", fontsize=14)
                # plt.xlabel("States", fontsize=14)
                # plt.xticks(np.arange(N_states + 1))
                plt.title("Upper and Lower bounds", fontsize=14)


                plt.subplot(122)
                if step >= 1:
                    norm_upper = np.linalg.norm(upper_list[-2] - upper_list[-1])
                    norm_list_upper.append(norm_upper)
                    plt.plot(norm_list_upper)
                    plt.title("Upper norm")
                plt.show()
                if step == total_steps - 1:
                    plt.savefig('pic1.png')
                # clear_output(True)
                # plt.figure(figsize=[16, 9])

                # assert not np.isnan(loss_history[-1])
                # plt.subplot(1, 2, 1)
                # plt.title("loss history")
                # # plt.plot(utils.smoothen(loss_history))
                # plt.plot(loss_history)
                # plt.grid()

                # plt.subplot(1, 2, 2)
                # plt.title("Grad norm history")
                # # plt.plot(utils.smoothen(grad_norm_history))
                # plt.plot(grad_norm_history)
                # plt.grid()
                # plt.show()


    return V_up, V_pi, X_samples.numpy(), upper_bound_sample


def plotBounds(V_up, V_pi, X_grid, X_data, ax, params):
    """
    INPUT:
    V_up - upper bounds regression: type = tensor
    X_grid - elements, which are estimated with V_up: type = ndarray
    X_data - elements, which we need to estimate with V_up: type = tensor
    params - dict of parameters
    """
    mpl.style.use('seaborn')
    neigh = NearestNeighbors(n_neighbors=5, algorithm='kd_tree', leaf_size=100, n_jobs=-1)
    neigh.fit(X_grid.tolist())
    _, idxes_neigh = neigh.kneighbors(X_data.numpy().tolist())
    V_up_data = V_up[torch.tensor(idxes_neigh)].mean(dim=-1).numpy()
    V_pi_data = V_pi
    N_states = X_data.shape[0]
    upper_plot = np.repeat(V_up_data.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
    lower_plot = np.repeat(V_pi_data.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
    states_plot = np.concatenate((np.arange(N_states).reshape(-1,1),
                                 (np.arange(N_states)+1).reshape(-1,1)), axis=1).reshape(-1)
    ax.fill_between(states_plot, lower_plot, upper_plot, alpha=0.4,
                edgecolor='green', facecolor='green', linestyle='-')
    major_ticksx = params['major_ticksx']
    major_ticksy = params['major_ticksy']
    ax.set_ylim(*params['y_lim'])
    ax.set_xticks(major_ticksx)
    ax.set_yticks(major_ticksy)
    ax.tick_params(axis="x", labelsize=params['tick_size'])
    ax.tick_params(axis="y", labelsize=params['tick_size'])