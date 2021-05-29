import numpy as np
import torch
import matplotlib.pyplot as plt
import acrobot
import tqdm
from IPython.display import clear_output
import time
import model as storage
import seaborn as sns
from tqdm import trange
from sklearn.neighbors import NearestNeighbors

def get_traj(n_samples, model, t=100):
    env = acrobot.AcrobotEnv(noise=True)
    states_list = []

    for k in tqdm.notebook.tqdm(range(n_samples)):
      s = env.reset()
      ep_num = 0

      while True:
          states_list.append(env.state)

          if isinstance(model, storage.Policy):
            action_probs = model(torch.FloatTensor(s))
            a = np.random.choice(3, p=action_probs.detach().numpy())

          elif isinstance(model, storage.DuelingDQN):
            a = model(torch.FloatTensor([s])).max(dim=1)[1]
          else:
            raise ValueError('Model is not supported.')
          
          new_s, r, done, _ = env.step(a)
          s = new_s
          ep_num += 1
          if done or ep_num >= t:
            break

    samples = np.array(states_list)

    return samples

def get_samples_from_traj(n_samples, model, t=50):

    env = acrobot.AcrobotEnv(noise=True)
    states_list = []

    for k in tqdm.notebook.tqdm(range(n_samples)):
      s = env.reset()
      ep_num = 0

      while True:
          if ep_num == t:
              states_list.append(env.state)
              break

          if isinstance(model, storage.Policy):
            action_probs = model(torch.FloatTensor(s))
            a = np.random.choice(3, p=action_probs.detach().numpy())

          elif isinstance(model, storage.DuelingDQN):
            a = model(torch.FloatTensor([s])).max(dim=1)[1]
          else:
            raise ValueError('Model is not supported.')

          new_s, r, done, _ = env.step(a)
          s = new_s
          ep_num += 1
          if done:
            break

    samples = np.array(states_list)

    rewards = np.zeros((samples.shape[0], 3))
    for num_sample, X in enumerate(samples):
      for action in range(3):
        env.reset(X)
        _, r, _, _ = env.step(action)
        rewards[num_sample, action] = r

    return samples, rewards

def get_rewards(samples):
    env = acrobot.AcrobotEnv(noise=True)
    
    rewards = np.zeros((samples.shape[0], 3))
    for num_sample, X in enumerate(samples):
      for action in range(3):
        env.reset(X)
        _, r, _, _ = env.step(action)
        rewards[num_sample, action] = r
    return rewards

def get_distributions(n_samples, model, plot=True):
    env = acrobot.AcrobotEnv(noise=True)
    total_states = np.zeros((n_samples, 50, 4))

    for k in tqdm.notebook.tqdm(range(n_samples)):
      s = env.reset()
      ep_num = 0
      while True:
          total_states[k, ep_num] = env.state
          
          if isinstance(model, storage.Policy):
            action_probs = model(torch.FloatTensor(s))
            a = np.random.choice(3, p=action_probs.detach().numpy())

          elif isinstance(model, storage.DuelingDQN):
            a = model(torch.FloatTensor([s])).max(dim=1)[1]
          else:
            raise ValueError('Model is not supported.')

          new_s, r, done, _ = env.step(a)
          s = new_s
          ep_num += 1
          if done or ep_num >= 50:
            break

    if plot:
        for t in np.arange(0, 50, 2):
          plt.figure(figsize=(20, 5))
          plt.subplot(141)
          count, bins, ignored = plt.hist(total_states[:, t, 0], 30, density=True)
          plt.title("theta_1, t={}".format(t))

          plt.subplot(142)
          count, bins, ignored = plt.hist(total_states[:, t, 1], 30, density=True)
          plt.title("theta_2, t={}".format(t))

          plt.subplot(143)
          count, bins, ignored = plt.hist(total_states[:, t, 2], 30, density=True)
          plt.title("theta_1_dot, t={}".format(t))

          plt.subplot(144)
          count, bins, ignored = plt.hist(total_states[:, t, 3], 30, density=True)
          plt.title("theta_2_dot, t={}".format(t))

          plt.savefig("distr_{}.png".format(t))
          plt.show()
          clear_output(True)
          time.sleep(0.1)

    return total_states

def is_terminal(s):
  return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)

def get_tensor(samples_traj):
    samples_tensor_traj = np.zeros((samples_traj.shape[0], 6))
    samples_tensor_traj[:, 0:4] = np.stack((np.cos(samples_traj[:, 0]), np.sin(samples_traj[:, 0]), 
                                np.cos(samples_traj[:, 1]), np.sin(samples_traj[:, 1]))).T
    samples_tensor_traj[:, 4:6] = samples_traj[:, -2:] 
    samples_tensor_traj = torch.tensor(samples_tensor_traj, dtype=torch.float32)
    
    return samples_tensor_traj

def get_continous_rewards(env, net, gamma=0.9):
    s = env.reset()
    llist = []
    cur_gamma = 1
    while True:
        a = net(torch.FloatTensor([s])).max(dim=1)[1]
        new_s, r, done, _ = env.step(a)
        s = new_s
        cur_gamma *= gamma
        state = env.state
        l = -env.LINK_LENGTH_1 - env.LINK_LENGTH_1*np.cos(state[0]) - env.LINK_LENGTH_2*np.cos(state[0] + state[1])
        llist.append(l)
        clear_output(True)
        plt.plot(llist)
        if done:
            break
    return llist

def MonteCarloEval(env, s0, model, n_samples=5, gamma = 0.9):
    average_reward = 0.
    total_states = []
    for k in range(n_samples):
        discounted_reward = 0.
        s = env.reset(torch.FloatTensor(s0))
        # s = env.reset()
        cur_gamma = 1.
        while True:
            if isinstance(model, storage.Policy):
              action_probs = model(torch.FloatTensor(s))
              a = np.random.choice(3, p=action_probs.detach().numpy())

            elif isinstance(model, storage.DuelingDQN):
              a = model(torch.FloatTensor([s])).max(dim=1)[1]
            else:
              raise ValueError('Model is not supported.')

            new_s, r, done, _ = env.step(a)
            discounted_reward += cur_gamma * r
            s = new_s
            cur_gamma *= gamma
            if done:
                break
        average_reward += discounted_reward

    return average_reward / n_samples


def getBounds(env, model, samples, V_pi, gamma, M1, M2, k, 
              total_steps=300, eval_freq = 1, save_freq=10, filename="plot_bounds.png"):
  sns.set(style="darkgrid")
  
  samples_tensor = get_tensor(samples)
  N = samples.shape[0]
  state_dim, n_actions = env.observation_space.shape, env.action_space.n

  #fit knn
  neigh = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', leaf_size=100, n_jobs=-1)
  neigh.fit(samples_tensor.numpy().tolist())

  # generate Y_states 
  # V_ns = torch.zeros((n_actions, N, M1))
  Y = torch.zeros((n_actions, N, M1, 6))

  for i in tqdm.notebook.tqdm(range(N)):
    for action in range(3):
      for j in range(M1):
        env.reset(samples[i])
        next_state, _, _, _ = env.step(action)
        #V_ns[action, i, j] = MonteCarloEval(env.state, policy_value, 8, gamma)
        next_state_tensor = torch.from_numpy(next_state).float()
        Y[action, i, j] = next_state_tensor

  _, idxes_neigh1 = neigh.kneighbors(torch.flatten(Y, end_dim=2).numpy().tolist())
  V_ns = V_pi[torch.tensor(idxes_neigh1)].squeeze().mean(dim=-1).reshape(n_actions, N, M1)
  V_mean = torch.mean(V_ns, dim=-1, keepdims=True)

  loss_history = []
  N = samples.shape[0]
  V_pi = V_pi.squeeze()

  rewards = get_rewards(samples)
  rewards = torch.tensor(rewards.reshape(n_actions, N, 1))

  V_up = (torch.zeros_like(V_pi) + torch.max(V_pi)).squeeze()
  upper_list = [V_up.detach().numpy()]
  norm_list_upper = []
  step = 0

  with trange(step, total_steps + 1) as progress_bar:
    for step in progress_bar:
      Y = torch.zeros((n_actions, N, M2, 6))
      for i in range(N):
        for j in range(M2):
          for action in range(3):
            env.reset(samples[i])
            next_state, _, _, _ = env.step(action)
            next_state_tensor = torch.from_numpy(next_state).float()
            Y[action, i, j] = next_state_tensor
                                      
      _, idxes_neigh2 = neigh.kneighbors(torch.flatten(Y, end_dim=2).numpy().tolist())
      
      V_k = V_up[torch.tensor(idxes_neigh2)].mean(dim=-1).reshape(n_actions, N, M2)
      V_pi_next = V_pi[torch.tensor(idxes_neigh2)].mean(dim=-1).reshape(n_actions, N, M2)

      V_up = (rewards + gamma * (V_k - V_pi_next + V_mean)).max(dim=0)[0].mean(dim=-1)
      upper_list.append(V_up.detach().numpy())
      
      if step % eval_freq == 0 and step >= 1:
        clear_output(True)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        norm_upper = np.linalg.norm(upper_list[-2] - upper_list[-1])
        norm_list_upper.append(norm_upper)
        plt.plot(norm_list_upper)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        plt.gcf().subplots_adjust(bottom=0.1, left = 0.2)
        plt.xlabel("Iteration Number", fontsize=20)
        plt.title("Upper norm", fontsize=20)
        plt.savefig(filename)
        plt.show()

      if step % save_freq == 0:
        np.savetxt("upper_bound.txt", V_up)

  return V_up

def plotBoundsRandomTraj(env, model, t, samples, V_up, gamma=0.9, filename="acrobot_traj.png"):
    traj = get_traj(1, model, t)
    samples_tensor = get_tensor(samples)
    traj_tensor = get_tensor(traj)

    k=2
    neigh = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', leaf_size=100, n_jobs=-1)
    neigh.fit(samples_tensor.numpy().tolist())
    _, idxes_neigh = neigh.kneighbors(traj_tensor.numpy().tolist())

    V_up_traj = V_up[torch.tensor(idxes_neigh)].mean(dim=-1)
    V_mont_traj = torch.tensor([MonteCarloEval(env, sample, model, 8, gamma) 
                                for sample in tqdm.notebook.tqdm(traj)])

    fig = plt.figure(figsize=(8, 7))
    sns.set(style="darkgrid")
    ax = fig.add_subplot(1, 1, 1)

    N_states = traj.shape[0]
    lower = V_mont_traj
    upper = V_up_traj

    upper_plot = np.repeat(upper.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
    lower_plot = np.repeat(lower.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
    states_plot = np.concatenate((np.arange(N_states).reshape(-1,1),
                              (np.arange(N_states)+1).reshape(-1,1)), axis=1).reshape(-1)
    plt.fill_between(states_plot, lower_plot, upper_plot, alpha=0.5,
              edgecolor='k', linestyle='-', label="V_up - V_pi")
    plt.legend(loc="upper left", fontsize=20)
    plt.xlabel("Trajectory States", fontsize=20)
    plt.plot(states_plot, upper_plot, states_plot, lower_plot, color='b')

    ax.set_xlim(0, 50)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    plt.gcf().subplots_adjust(bottom=0.1, left = 0.2)

    plt.savefig(filename)

def compare_approx(env, model, t, samples, V_pi, gamma=0.9, filename="acrobot_traj_mc_knn.png"):
    # generate random trajectory
    traj = get_traj(1, model, t)
    samples_tensor = get_tensor(samples)
    traj_tensor = get_tensor(traj)

    k=2
    neigh = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', leaf_size=100, n_jobs=-1)
    neigh.fit(samples_tensor.numpy().tolist())
    _, idxes_neigh = neigh.kneighbors(traj_tensor.numpy().tolist())

    V_pi_traj = V_pi[torch.tensor(idxes_neigh)].mean(dim=-1)
    V_mont_traj = torch.tensor([MonteCarloEval(env, sample, model, 8, gamma) 
                                for sample in tqdm.notebook.tqdm(traj)])
    fig = plt.figure(figsize=(8, 7))
    sns.set(style="darkgrid")
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(V_mont_traj, '.-', label='Monte-Carlo')
    plt.plot(V_pi_traj, '.-', label='KNN')
    plt.legend(fontsize=25)
    plt.ylabel("Value", fontsize=20)
    plt.xlabel("Trajectory States", fontsize=20)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    plt.gcf().subplots_adjust(bottom=0.1, left = 0.2)

    plt.savefig(filename)
