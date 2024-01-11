import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

from rl_playground.utils.single_integrator import *
import rl_playground.utils.plot_utils as plut


''' Basic implementation of Reinforce algorithm with discrete action space, 
    i.e. policy gradient using empirical return (or cost-to-go) as weight for each trajectory.
    Implementation based on PyTorch for representing the policy as a NN, and
    tested on a simple non-convex 1D single integrator problem.
'''
def dynamic_saturated(x, u):
    x_next = dynamic(x, u)
    x_next = np.minimum(x_max, np.maximum(x_next, x_min))
    return x_next

def cost_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# make function to compute action distribution
def get_policy(x):
    logits = logits_net(x)
    return Categorical(logits=logits)

# make action selection function (outputs int actions, sampled from policy)
def sample_control(x):
    return get_policy(torch.as_tensor(x, dtype=torch.float32)).sample().item()

def get_max_likelihood_control(x):
    logits = logits_net(torch.as_tensor(np.array([x]), dtype=torch.float32))
    return np.argmax(logits.detach().numpy())

def get_mean_control(x):
    actions = np.arange(0, n_u)
    probs = get_policy(torch.as_tensor(np.array([x]), dtype=torch.float32)).probs
    return probs.detach().numpy().dot(actions)

def discr_to_cont_control(u_discr):
    return u_min + u_discr*(u_max-u_min)/(n_u-1)

# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(x, u, weights):
    logp = get_policy(x).log_prob(u)
    return (logp * weights).mean()

def plot_actor(show=True):
    plt.figure()
    plt.plot(X_grid, running_cost, label="cost", alpha=0.5)

    # U_ml_discr = [get_max_likelihood_control(x) for x in X_grid]
    # U_ml_cont = discr_to_cont_control(U_ml_discr)
    # plt.plot(X_grid, U_ml_cont, 'o ', label="Max Likelihood Control")
    
    # U_mean_discr = [get_mean_control(x) for x in X_grid]
    # U_mean_cont = discr_to_cont_control(U_mean_discr)
    # plt.plot(X_grid, U_mean_cont, 'x ', label="Mean Control")

    U_cont = discr_to_cont_control(np.arange(n_u))
    for i in range(X_grid.shape[0]):
        x = torch.as_tensor(X_grid[i,:], dtype=torch.float32)
        probs = get_policy(x).probs.detach().numpy()
        probs /= np.max(probs)
        for u in range(n_u):
            plt.plot(X_grid[i], U_cont[u], 'ro ', alpha=probs[u])
    
    plt.title("Actor")
    plt.xlabel("State x")
    if(show): 
        plt.legend()
        plt.show()



if __name__=='__main__':
    hidden_sizes=[8,8,8]
    lr=1e-2
    epochs=20
    batch_size=5000
    network_updates = 10

    N_PLOT = 10 # show plots every N_PLOT EPOCHS

    n_x = 1
    n_u = 2
    N = 20
    x_min, x_max = np.array([-2.2]), np.array([2.0])
    u_min, u_max = np.array([-1.0]), np.array([1.0])
    
    N_grid = 100
    X_grid = np.linspace(x_min, x_max, N_grid).reshape((N_grid,n_x))
    # U_grid = np.linspace(u_min, u_max, N_grid).reshape((N_grid,n_u))
    running_cost = [cost(np.array([x]), 0) for x in X_grid]

    # make policy network and optimizer
    logits_net = mlp(sizes=[n_x]+hidden_sizes+[n_u]) #,  activation=nn.ReLU)
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_states = []       # for states
        batch_ctrl = []         # for control inputs
        batch_weights = []      # for cost-to-go weighting in policy gradient
        batch_ctg = []          # for measuring episode cost-to-go

        # reset episode-specific variables
        t, ep_cst, x = 0, [], np.random.uniform(x_min, x_max, size=(n_x))

        # collect experience by acting in the environment with current policy
        while True:
            # save x
            batch_states.append(x)
            u_discr = sample_control(x)
            u_cont = discr_to_cont_control(u_discr)
            cst = cost(x, u_cont)
            x = dynamic_saturated(x, u_cont)

            # save action, reward
            batch_ctrl.append(u_discr)
            ep_cst.append(cst)
            t+=1
            if t==N:
                # if episode is over, record info about episode
                ep_ctg = sum(ep_cst)
                batch_ctg.append(ep_ctg)
                # the weight for each logprob(a|s) is R(tau)
                # batch_weights += [ep_ctg] * N
                batch_weights += list(cost_to_go(ep_cst))
                # reset episode-specific variables
                t, ep_cst, x = 0, [], np.random.uniform(x_min, x_max, size=(n_x))
                # end experience loop if we have enough of it
                if len(batch_states) > batch_size:
                    break

        # take policy gradient update step
        for j in range(network_updates):
            optimizer.zero_grad()
            batch_loss = compute_loss(x=torch.as_tensor(batch_states, dtype=torch.float32),
                                    u=torch.as_tensor(batch_ctrl, dtype=torch.int32),
                                    weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                    )
            batch_loss.backward()
            optimizer.step()
        return batch_loss, batch_ctg, batch_states, batch_weights

    # training loop
    for i in range(epochs):
        batch_loss, batch_ctg, batch_states, batch_weights = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t avg cost per step: %.3f'%
                (i, batch_loss, np.mean(batch_ctg)/N))
        
        if((i+1)%N_PLOT==0):
            V = np.zeros(N_grid)
            for (i, x_init) in enumerate(X_grid):
                x = np.copy(x_init)
                for t in range(N):
                    u_discr = get_max_likelihood_control(x)
                    u_cont = discr_to_cont_control(u_discr)
                    V[i] += cost(x, u_cont)
                    x = dynamic_saturated(x, u_cont)
                V[i] += cost(x, np.zeros(n_u))
            print("Avg cost/step of max-likel. ctrl: %.3f"%(np.mean(V)/(N+1)))

            plot_actor(show=False)
            # plt.plot(batch_states, (1/N)*np.array(batch_weights), 'x ', label="Cost-to-go", alpha=0.1)
            plt.plot(X_grid, V/(N+1), label="Avg value/step max-lik.", alpha=0.5)
            plt.legend()
            # plt.show()
    
    # EVALUATE STOCHASTIC POLICY BY MONTE-CARLO SAMPLING
    # N_SAMPLES = 100
    # V = np.zeros(N_grid)
    # for (i, x_init) in enumerate(X_grid):
    #     for j in range(N_SAMPLES):
    #         x = np.array([x_init])
    #         for t in range(N):
    #             u_discr = sample_control(x)
    #             u_cont = discr_to_cont_control(u_discr)
    #             V[i] += cost(x, u_cont)
    #             x = dynamic(x, u_cont)
    #         V[i] += cost(x, np.zeros(n_u))
    #     V[i] /= N_SAMPLES
    # print("Avg cost/step of stochastic policy: %.3f"%(np.mean(V)/(N+1)))
    # plot_actor(show=False)
    # # plt.plot(batch_states, (1/N)*np.array(batch_weights), 'x ', label="Cost-to-go", alpha=0.1)
    # plt.plot(X_grid, V/(N+1), label="Avg value/step stoch.", alpha=0.5)
    # plt.legend()
    # plt.show()
