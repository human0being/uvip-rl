import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

def getDiscrStationaryBounds(policy, rewards, P, p_ksi, Y_states, gamma, T = 1000, eps = 0.1, plotNorm=True, computeEmp=True):
    """
    Get Upper Bound for Stationary Value Function in a case of Discrete states and actions:
    INPUT:
    policy - matrix of probabilities to accept action a under the condition of state s size of [N_states, N_actions],
             where N_states - number of states, N_actions - number of actions;
    rewards - reward matrix size of [N_states, N_actions];
    Y_states - matrix of next states of size [N_states, N_actions, N_br], 
               where N_br - number of ksi;
    P - dynamics of the system size of [N_states, N_states, N_actions];
    p_ksi - vector of probabilities to accept ksi, which is to be in one of states [0, 1, ..., N_br - 1];
    gamma - discounting factor;
    T - number of iterations for computing next states, default T = 1000;
    eps - accuracy of convergence of the upper bound in the norm, default eps = 0.1
    
    OUTPUT:
    V_star - vector of N_states, computed value function for a given policy using Bellman principle; (Lower bound)
    upper_list[-1] - vector of N_states, upper bound for V_star;
    norm_list_upper - list of norms between upper bounds, shows convergence;   
    """
    sns.set(style="darkgrid")#, font_scale = 2.0)
    
    N_states = Y_states.shape[0]
    N_actions = Y_states.shape[1]
    N_br = Y_states.shape[2]
    np.random.seed(None)
    
    # compute V via Bellman principle
    b = np.sum(rewards*policy, axis=1)
    A = np.einsum('km,nkm->nk', policy, P).T
    A = np.eye(N_states) - gamma*A

    V_star = np.linalg.solve(A, b)
    
    # compute martingales
    M = np.zeros_like(Y_states, dtype=float)
    
    if not computeEmp:
        M =  V_star[Y_states] - np.sum(V_star[Y_states]*p_ksi, axis=2, keepdims=True)
    else:
        for i in range(N_states):
            for k in range(N_actions):
                ksi = np.random.choice(a=np.arange(N_br), size=T, p=p_ksi)
                for j in range(N_br):
                    M[i, k, j] =  V_star[Y_states[i, k, j]] - np.sum(V_star[Y_states[i, k, ksi]])/T
                
    # compute bounds
    states = np.arange(N_states)
    upper_list = [V_star]
    #lower_list = [V_star]
    
    norm_list_upper = []
    #norm_list_lower = []

    norm_theta = 0
    
    while True:
    
        upper_average = np.zeros(N_states)
        #lower_average = np.zeros(N_states)
        
        for j in range(T):
            ksi = np.random.choice(a=np.arange(N_br), size=N_states, p=p_ksi)
            next_states = Y_states[states, :, ksi]

            upper_average += np.max(rewards + gamma*(upper_list[-1][next_states] - M[states, :, ksi]), axis=1)
            #lower_average += np.einsum("nk, nk-> n", rewards + gamma*(lower_list[-1][next_states] - M[states, :, ksi]), policy)

        upper_average /= T
        #lower_average /= T
        
        upper_list.append(upper_average)
        #lower_list.append(lower_average)

        if len(upper_list) > 2:
            norm_upper = np.linalg.norm(upper_list[-2] - upper_list[-1])
            #norm_lower = np.linalg.norm(lower_list[-2] - lower_list[-1])
            
            if norm_upper < eps:
                print("Norm upper:", norm_upper)
                #print("Norm lower:", norm_lower)
                break
            
            norm_list_upper.append(norm_upper)
            #norm_list_lower.append(norm_lower)
            
            if plotNorm:
                clear_output(wait=True)
                fig = plt.figure(figsize=(8, 7))
                ax = fig.add_subplot(1, 1, 1)
                #plt.subplot(121)
                plt.plot(norm_list_upper)
                plt.title("Upper norm")

                #plt.subplot(122)
                #plt.plot(norm_list_lower)
                #plt.title("Lower norm")
                plt.show()
            
    return V_star, upper_list[-1], norm_list_upper #, norm_list_lower, lower_list[-1]
    
def plotBounds(bounds, iter_num="0", path="./pics/", save=False, legend_location="upper left", opt_v=None):
    """
    INPUT:
    bounds - dictionary of {"policy": "V", "upper"};
    """  
    os.makedirs(path, exist_ok=True) 
    
    fig = plt.figure(figsize=(8, 7))
    sns.set(style="darkgrid")#, font_scale = 2.0)
    ax = fig.add_subplot(1, 1, 1)
    
    for policy_name in bounds.keys():
        
        lower = bounds[policy_name]["V"]
        upper = bounds[policy_name]["upper"]
        #lower = bounds[policy_name]["lower"]
        N_states = len(upper)
        x = np.arange(N_states + 1)
        
        upper_plot = np.repeat(upper.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
        V_plot = np.repeat(lower.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
        states_plot = np.concatenate((np.arange(N_states).reshape(-1,1), 
                                      (np.arange(N_states)+1).reshape(-1,1)), axis=1).reshape(-1)
        #if opt_v is not None:
            #opt_v_plot = np.repeat(opt_v.reshape(-1, 1), repeats=2, axis=1).reshape(-1)
            #dict_ = {"opt": opt_v_plot, "x": states_plot}
            #dict_ = pd.DataFrame(data=dict_)
            #sns.lineplot(data=dict_, x='x', y='opt')
            #plt.plot(states_plot, opt_v_plot)
            
        plt.fill_between(states_plot, V_plot, upper_plot, alpha=0.5, 
                        edgecolor='b', facecolor='purple', linestyle='-')#, label=policy_name)
        plt.plot(states_plot, V_plot, color='purple', linewidth=2)
        plt.plot(states_plot, upper_plot, color='purple', linewidth=2)

    #plt.legend(loc=legend_location, fontsize=14)
    #plt.xlabel("states")
    #plt.ylabel("value")
    #plt.xticks(x)
    ax.tick_params(axis="x", labelsize=35)
    ax.tick_params(axis="y", labelsize=35)
    #ax.set_ylim(-0.1,8.0)
    #ax.set_ylim(30,130)
    plt.tight_layout()
    #plt.ylim(-0.1,7.0)
    #plt.title("Upper and Lower bounds", fontsize=14)
    if save:
        plt.savefig((path + "file_{}.png").format(iter_num), pi=fig.dpi)
        plt.close()
    else:
        plt.show()
        
# making transition matrix for Garnet
def transition_matrix(N_states, N_actions, N_br, random_state=None):
    P = np.zeros((N_states, N_states, N_actions))
    for i in range(N_states):
        for j in range(N_actions):
            if random_state is not None:
                np.random.seed(i + j + random_state)
            random_states = np.random.choice(np.arange(N_states), size=N_br, replace=False)
            P[random_states, i, j] = 1/N_br
    return P

# making rewards matrix for Garnet
def get_reward(N_states, N_actions, random_state=None):
    np.random.seed(random_state)
    r_sa = np.random.random(size=(N_states, N_actions))
    random_states = np.random.choice(np.arange(N_states), N_states)
    random_actions = np.random.choice(np.arange(N_actions), N_states)
    r_sa[random_states, random_actions] *= 20
    return r_sa

# get environment matrix
def get_dynamics(env):
    p_next_state = np.zeros((env.observation_space.n, env.observation_space.n, env.action_space.n))

    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            for next_state in range(env.observation_space.n):
                for i in range(len(env.P[state][action])):
                    if next_state == env.P[state][action][i][1]:
                        p_next_state[next_state, state, action] +=\
                            env.P[state][action][i][0]
    return p_next_state

def perform_value_iteration(P, r_sa, Y_states, p_ksi, T, gamma, bounds, eps=.00001, bounds_eps = 0.1, 
                            plotNorm=False, compEmp=True, path='./pics/', save=True):
    """
    Input:
    P - dynamics of the system size of [N_states, N_states, N_actions];
    r_sa - reward matrix size of [N_states, N_actions];
    Y_states - matrix of next states of size [N_states, N_actions, N_br], 
               where N_br - number of ksi;
    p_ksi - vector of probabilities to accept ksi, which is to be in one of states [0, 1, ..., N_br - 1];
    T - number of iterations for computing next states;
    gamma - discounting factor;
    bounds - dict for storing V and V^up;
    eps - accuracy of Value iteration convergence, default eps=.00001;
    bounds_eps - accuracy of convergence of the upper bound in the norm, default bounds_eps = 0.1;
    plotNorm - plot the convergence of upper bound, bool;
    compEmp - use Monte-Carlo for expectations computation, bool;
    path - path of saving upper and lower bounds plots;
    save - save upper and lower bounds plots, bool; 
    
    Output:
    policy_determ - deterministic policy from Value Iteration procedure, size [N_states, N_actions];
    """
    
    N_states = P.shape[0]
    N_actions = P.shape[2]
    Q_prev = np.random.randn(N_states, N_actions)
    policy_prev = np.zeros((N_states, N_actions))
    norm_list = []
    j = 0
    T = 1000        #M1=M2=T

    while True:
        Q = r_sa + gamma * np.einsum("n, nkm -> km", np.max(Q_prev, axis=1), P)

        if j > 2:
            norm_list.append(np.linalg.norm(Q_prev - Q))
            if norm_list[-1] < eps:
                break
        Q_prev = Q.copy()

        policy_determ = np.zeros((N_states, N_actions))
        for i in range(N_states):
            policy_determ[i, np.argmax(Q, axis=1)[i]] = 1

        if j in [0, 5, 15]:
            V, upper, _  = getDiscrStationaryBounds(policy_determ, r_sa, 
                                            P, np.array(p_ksi), Y_states, gamma, int(T), 
                                            bounds_eps, plotNorm, compEmp)
            bounds["policy"]["V"] = V
            bounds["policy"]["upper"] = upper

            clear_output(wait=True)
            plotBounds(bounds, j, path, save)
            bounds_eps /= 1.4

        policy_prev = policy_determ.copy()
        j += 1
        
    return policy_determ


# tabular reinforce
def update_policy_tabular(env, policy, theta, rewards, trajectory, lr, gamma):
    T = len(trajectory)
    theta_grad = np.zeros_like(theta)
    for t in range(T):
        G = gamma**(T-t-1)

        s, a = trajectory[t]
        theta_grad[a+env.action_space.n*s] += lr*G*(1-policy[s, a])

    theta += theta_grad
    policy = make_policy(env, theta)
    return theta, policy

def make_policy(env, theta):
    pi = np.exp(theta).reshape(env.observation_space.n, env.action_space.n)
    pi = pi/np.sum(pi, axis=1, keepdims=True)
    return pi