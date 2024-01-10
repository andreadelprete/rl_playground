import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

from rl_playground.utils.single_integrator import *
import rl_playground.utils.plot_utils as plut


''' Script to debug a basic implementation of Reinforce algorithm with discrete action space, 
    i.e. policy gradient using empirical return as weight for each trajectory.
    Implementation based on PyTorch for representing the policy as a NN, and
    tested on a simple non-convex 1D single integrator problem.
'''
def cost(x, u):
    # return x[0]**2
    x = x[0]
    c = 0.0*u*u + (x-1.9)*(x-1.0)*(x-0.6)*(x+0.5)*(x+1.2)*(x+2.1)
    return c[0]

def positive_cost(x, u):
    return cost(x, u) #+ 7.0

def cost_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


def dynamic_saturated(x, u):
    x_next = dynamic(x, u)
    x_next = np.minimum(x_max, np.maximum(x_next, x_min))
    return x_next

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
    assert(np.all(u_discr>=0) and np.all(u_discr<n_u))
    return u_min + u_discr*(u_max-u_min)/(n_u-1)

# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(x, u, weights):
    # logits = logits_net(x)[np.arange(weights.shape[0], dtype=int), u]
    # return (logits*weights).mean()
    logp = get_policy(x).log_prob(u)
    return (logp * weights).mean()

def plot_episode_data(X, U, W, show=True):
    # fig, ax = plut.create_empty_figure(n_u)
    W = W/N
    # for i in range(n_u):
    #     ax[i].plot(X_grid, running_cost, label="cost", alpha=0.5)
    #     ax[i].set_title("U="+str(i))
    # for i in range(X.shape[0]):
    #     ax[U[i]].plot(X[i], W[i], 'ro ', alpha=0.5)
    plt.figure()
    plt.plot(X_grid, running_cost, label="cost", alpha=0.5)
    for i in range(n_u):
        ind = np.where(U==i)[0]
        plt.plot(X[ind], W[ind], 'x ', label="u="+str(i), alpha=0.5)
    plt.xlabel("State x")
    if(show): 
        plt.legend()
        plt.show()

def plot_actor(show=True):
    plt.figure()
    plt.plot(X_grid, running_cost, label="cost", alpha=0.5)

    # U_ml_discr = [get_max_likelihood_control(x) for x in X_grid]
    # U_ml_cont = discr_to_cont_control(U_ml_discr)
    # plt.plot(X_grid, U_ml_cont, 'o ', label="Max Likelihood Control")

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
    hidden_sizes=[8, 8, 8]
    lr=1e-2
    epochs=200
    batch_size=500
    
    N_PLOT = 10 # show plots every N_PLOT EPOCHS
    PLOT_PROB = 0 # enable plots of probabilities of different actions

    n_x = 1
    n_u = 2
    N = 30
    # x_min, x_max = np.array([-2.2]), np.array([2.0])
    x_min, x_max = np.array([-2.2]), np.array([2.0])
    u_min, u_max = np.array([-1.0]), np.array([1.0])
    
    N_grid = 100
    X_grid = np.linspace(x_min, x_max, N_grid).reshape((N_grid,n_x))
    running_cost = [cost(np.array([x]), 0) for x in X_grid]

    # make policy network and optimizer
    logits_net = mlp(sizes=[n_x]+hidden_sizes+[n_u]) #,  activation=nn.ReLU)
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # training loop
    for i in range(epochs):
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
            cst = positive_cost(x, u_cont)
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
                # batch_weights += [ep_ctg]*N
                batch_weights += list(cost_to_go(ep_cst))
                # reset episode-specific variables
                t, ep_cst, x = 0, [], np.random.uniform(x_min, x_max, size=(n_x))
                # end experience loop if we have enough of it
                if len(batch_states) >= batch_size:
                    break

        # print("Finished running", batch_size/N, "episodes with cost-to-go's ranging btw", 
        #       np.min(batch_ctg), np.max(batch_ctg))

       
        X_torch = torch.as_tensor(batch_states, dtype=torch.float32)
        U_torch = torch.as_tensor(batch_ctrl, dtype=torch.int32)
        W_torch = torch.as_tensor(batch_weights, dtype=torch.float32)
        X_np = np.array(batch_states)
        U_np = np.array(batch_ctrl, dtype=int)
        W_np = np.array(batch_weights)

        if(PLOT_PROB):    
            ind_W = np.argsort(W_np)
            W_sorted = W_np[ind_W]
            rn = np.arange(W_np.shape[0], dtype=int)
            pi = get_policy(X_torch)
            # prob_pre = np.array([pi.probs[i, batch_ctrl[i]].item() for i in rn])
            prob_pre = pi.probs[rn, U_torch].detach().numpy()
            logprob_pre = pi.log_prob(U_torch).detach().numpy()
            # logits_pre = logits_net(X_torch)[rn, U_torch].detach().numpy()
            weighted_logp_pre  = logprob_pre  * W_np
            ind_logp = np.argsort(weighted_logp_pre)
            weighted_logp_pre  = weighted_logp_pre[ind_logp]
        
        # take a single policy gradient update step
        batch_loss_pre = compute_loss(x=X_torch, u=U_torch, weights=W_torch)
        for j in range(1):
            optimizer.zero_grad()
            batch_loss = compute_loss(x=X_torch, u=U_torch, weights=W_torch)
            batch_loss.backward()
            optimizer.step()
            # print('\t\t\t inner iter %d loss: %.3f'%(j, batch_loss))
        batch_loss_post = compute_loss(x=X_torch, u=U_torch, weights=W_torch)

        print('epoch: %3d \t loss: %.3f \t delta loss: %.3f \t avg cost per step: %.3f'%
                (i, batch_loss_pre, batch_loss_post-batch_loss_pre, np.mean(batch_ctg)/N))
        
        # plot_episode_data(X_np, U_np, W_np, show=True)

        if(PLOT_PROB):
            pi = get_policy(X_torch)
            # prob_post = np.array([pi.probs[i, batch_ctrl[i]].item() for i in rn])
            prob_post = pi.probs[rn, U_torch].detach().numpy()
            logprob_post = pi.log_prob(U_torch).detach().numpy()
            # logits_post = logits_net(X_torch)[rn, U_torch].detach().numpy()
            # delta_logits  = logits_pre[ind_W] - logits_post[ind_W]
            weighted_logp_post = logprob_post * W_np
            weighted_logp_post = weighted_logp_post[ind_logp]
            
            delta_logprob    = logprob_post - logprob_pre
            delta_prob       = prob_post - prob_pre
            delta_weighted_logp = weighted_logp_post-weighted_logp_pre

            # plt.figure()
            # plt.plot(W_sorted/np.max(W_sorted), 'o ', label="Normalized weights")
            # plt.plot(delta_prob[ind_W]/np.max(delta_prob), 'x ', label="Delta prob")
            # # plt.plot(delta_logits/np.max(delta_logits), 'x ', label="Delta logits")
            # plt.legend()

            fig, ax = plut.create_empty_figure(3)
            ax[0].plot(W_np[ind_logp], 'o ', label="W_torch")
            ax[0].legend()
            ax[1].plot(logprob_pre[ind_logp], 'x ', label="Log Prob pre")
            # ax[1].plot(logprob_post[ind_logp], 'o ', label="Log Prob post")
            ax[1].plot(delta_logprob[ind_logp], 'o ', label="Delta Log Prob")
            ax[1].legend()
            ax[2].plot(prob_pre[ind_logp], 'x ', label="Prob pre")
            # ax[2].plot(prob_post[ind_logp], 'o ', label="Prob post")
            ax[2].plot(delta_prob[ind_logp], 'o ', label="Delta Prob")
            ax[2].legend()

            fig, ax = plut.create_empty_figure(3)
            ax[0].plot(weighted_logp_pre,  label="weighted logp pre",  alpha=0.5)
            ax[0].plot(weighted_logp_post, 'x ', label="weighted logp post", alpha=0.5)
            ax[0].plot([0, rn[-1]], 2*[np.mean(weighted_logp_pre)],  ':', label="Avg pre",  alpha=0.5)
            ax[0].plot([0, rn[-1]], 2*[np.mean(weighted_logp_post)], ':', label="Avg post", alpha=0.5)
            ax[0].legend()
            ax[1].plot(delta_weighted_logp, 'x ', label="delta weighted logp")
            ax[1].legend()
            ax[2].plot(np.sort(delta_weighted_logp), 'x ', label="sorted delta weighted logp")
            ax[2].legend()
            plt.show()

            

        if((i+1)%N_PLOT==0):
            V = np.zeros(N_grid)
            for (i, x_init) in enumerate(X_grid):
                # print("\nStarting simulation")
                x = np.copy(x_init)
                for t in range(N):
                    u_discr = get_max_likelihood_control(x)
                    u_cont = discr_to_cont_control(u_discr)
                    # print("t=", t, "x=", x, "u=", u_cont, "cost=", cost(x, u_cont))
                    V[i] += cost(x, u_cont)
                    x = dynamic_saturated(x, u_cont)
                V[i] += cost(x, np.zeros(n_u))
            print("Avg cost/step of max-likel. ctrl: %.3f"%(np.mean(V)/(N+1)))

            plot_actor(show=False)
            # plt.plot(batch_states, (1/N)*np.array(batch_weights), 'x ', label="Cost-to-go", alpha=0.1)
            plt.plot(X_grid, V/(N+1), label="Avg value/step max-lik.", alpha=0.5)
            plt.legend()
            plt.show()

            # import time
            # time.sleep(1)
    