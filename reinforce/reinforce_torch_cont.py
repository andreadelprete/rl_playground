import torch
import torch.nn as nn
# from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

from rl_playground.utils.single_integrator import *
import rl_playground.utils.plot_utils as plut


''' Basic implementation of Reinforce algorithm with continuous action space, 
    i.e. policy gradient using empirical return as weight for each trajectory.
    Implementation based on PyTorch for representing the policy as a NN, and
    tested on a simple non-convex 1D single integrator problem.

    Does not work well in practice, even though the implementation seems correct.
'''
def cost(x, u):
    # return x[0]**2
    x = x[0]
    c = 0.0*u*u + (x-1.9)*(x-1.0)*(x-0.6)*(x+0.5)*(x+1.2)*(x+2.1)
    return c[0]

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
    mu_sigma = policy_net(x)
    if(len(mu_sigma.shape)==1):
        mu, sigma = mu_sigma[0:n_u], torch.abs(mu_sigma[n_u:])
    else:
        mu, sigma = mu_sigma[:,0:n_u], torch.abs(mu_sigma[:,n_u:])
    return Normal(loc=mu, scale=sigma)

# make action selection function (outputs int actions, sampled from policy)
def sample_control(x):
    return get_policy(torch.as_tensor(x, dtype=torch.float32)).sample().numpy()

def get_max_likelihood_control(x):
    mu_sigma = policy_net(torch.as_tensor(x, dtype=torch.float32))
    mu_sigma = mu_sigma.detach().numpy()
    return mu_sigma[:n_u]

def get_control_stddev(x):
    mu_sigma = policy_net(torch.as_tensor(x, dtype=torch.float32))
    mu_sigma = mu_sigma.detach().numpy()
    return mu_sigma[n_u:]

# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(x, u, weights):
    # logp = get_policy(x).log_prob(u)
    pi = get_policy(x)
    logp_ind = pi.log_prob(u)
    logp = logp_ind.sum(axis=-1)
    return (logp * weights).mean()

def plot_actor(show=True):
    plt.figure()
    plt.plot(X_grid, running_cost, label="cost", alpha=0.5)

    U_ml = np.array([get_max_likelihood_control(x) for x in X_grid])
    plt.plot(X_grid, U_ml, 'ro', label="Max Likelihood Control")

    U_sigma = np.array([get_control_stddev(x) for x in X_grid])
    plt.plot(X_grid, U_ml+U_sigma, 'rx ', label="Control+Sigma", alpha=0.3)
    plt.plot(X_grid, U_ml-U_sigma, 'rx ', label="Control+Sigma", alpha=0.3)
    
    plt.title("Actor")
    plt.xlabel("State x")
    if(show): 
        plt.legend()
        plt.show()



if __name__=='__main__':
    hidden_sizes=[8,8,8]
    lr=1e-2
    max_epochs=400
    batch_size=500

    n_x = 1
    n_u = 1
    N = 30
    x_min, x_max = np.array([-2.2]), np.array([2.0])
    u_min, u_max = np.array([-2.0]), np.array([2.0])
    
    N_grid = 100
    X_grid = np.linspace(x_min, x_max, N_grid).reshape((N_grid,n_x))
    # U_grid = jnp.linspace(u_min, u_max, N_grid).reshape((N_grid,n_u))
    running_cost = [cost(np.array([x]), 0) for x in X_grid]

    # make policy network and optimizer
    policy_net = mlp(sizes=[n_x]+hidden_sizes+[2*n_u],  activation=nn.ReLU)
    optimizer = Adam(policy_net.parameters(), lr=lr)

    # training loop
    for i in range(max_epochs):
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
            u = sample_control(x)
            cst = cost(x, u)
            x = dynamic_saturated(x, u)
            # save action, reward
            batch_ctrl.append(u)
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

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(x=torch.as_tensor(batch_states, dtype=torch.float32),
                                  u=torch.as_tensor(batch_ctrl, dtype=torch.float32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()

        print('epoch: %3d \t loss: %.3f \t avg cost per step: %.3f'%
                (i, batch_loss, np.mean(batch_ctg)/N))
        
        if((i+1)%20==0):
            V = np.zeros(N_grid)
            for (i, x_init) in enumerate(X_grid):
                x = np.copy(x_init)
                for t in range(N):
                    u = get_max_likelihood_control(x)
                    V[i] += cost(x, u)
                    x = dynamic(x, u)
                V[i] += cost(x, np.zeros(n_u))
            print("Avg cost/step of max-likel. ctrl: %.3f"%(np.mean(V)/(N+1)))

            plot_actor(show=False)
            # plt.plot(batch_states, (1/N)*np.array(batch_weights), 'x ', label="Cost-to-go", alpha=0.1)
            plt.plot(X_grid, V/(N+1), label="Avg value/step max-lik.", alpha=0.5)
            plt.legend()
            plt.show()

            # print("Stop training? [y/n]")
            # ans = input()
            # if(ans=='y'):
                # break
    