import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

from rl_playground.utils.single_integrator import *
import rl_playground.utils.plot_utils as plut


''' Basic implementation of Reinforce algorithm with discrete action space, 
    i.e. policy gradient using empirical return as weight for each trajectory.
    Implementation based on a tabular representation of the policy, and
    tested on a simple non-convex 1D single integrator problem.
'''

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

def logits_net(x):
    x_discr = ((x-x_min) / x_step).astype(int)[0]
    return logits[x_discr]

# make function to compute action distribution
def get_policy(x):
    logits_x = torch.as_tensor(logits_net(x), dtype=torch.float32)
    return Categorical(logits=logits_x)

# make action selection function (outputs int actions, sampled from policy)
def sample_control(x):
    return get_policy(x).sample().detach().numpy().reshape(1)

def get_max_likelihood_control(x):
    logits_x = logits_net(x)
    return np.argmax(logits_x)

def discr_to_cont_control(u_discr):
    return u_min + u_discr*(u_max-u_min)/(n_u-1)

# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(x, u, weights):
    logp = []
    for i in range(x.shape[0]):
        pi = get_policy(x[i,:])
        logp.append(pi.log_prob(torch.as_tensor(u[i,:], dtype=torch.int32)))
    # logp = get_policy(x).log_prob(torch.as_tensor(u, dtype=torch.int32))
    logp = torch.as_tensor(logp)
    return (logp * torch.as_tensor(weights)).mean()

# compute loss function gradient
def compute_loss_grad(x, u, weight):
    logits_x = torch.tensor(logits_net(x), dtype=torch.float32, requires_grad=True)
    pi = Categorical(logits=logits_x)
    logp = weight * pi.log_prob(torch.as_tensor(u, dtype=torch.int32))
    logp.backward()
    return logits_x.grad


def plot_actor(show=True):
    plt.figure()
    plt.plot(X_grid, running_cost, label="cost", alpha=0.5)

    U_cont = discr_to_cont_control(np.arange(n_u))
    for i in range(X_grid.shape[0]):
        x = X_grid[i,:]
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
    lr=3e-3
    epochs=20
    batch_size=5000
    baseline = 0 # remove a baseline from the reward to go?

    n_x = 1
    n_u = 2
    N = 20
    x_min, x_max = np.array([-2.2]), np.array([2.0])
    u_min, u_max = np.array([-1.0]), np.array([1.0])
    
    N_grid = 100
    x_step = (x_max - x_min) / (N_grid-1)
    X_grid = np.linspace(x_min, x_max, N_grid).reshape((N_grid, n_x))
    running_cost = [cost(X_grid[i,:], np.zeros(1)) for i in range(X_grid.shape[0])]

    # make tabular policy
    logits = [np.ones(n_u) for i in range(N_grid)]

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
                rtgs = list(cost_to_go(ep_cst))
                if(baseline):
                    # use as baseline the cost of the current state times the number of remaining time steps in the episode
                    for j in range(len(rtgs)):
                        rtgs[-1-j] -= (1+j)*cost(batch_states[-1-j], np.zeros(1))
                batch_weights += rtgs
                # reset episode-specific variables
                t, ep_cst, x = 0, [], np.random.uniform(x_min, x_max, size=(n_x))
                # end experience loop if we have enough of it
                if len(batch_states) >= batch_size:
                    break

        # print("Finished running", batch_size/N, "episodes with cost-to-go's ranging btw", 
        #       np.min(batch_ctg), np.max(batch_ctg))

        # W = np.array(batch_weights)
        # ind_W = np.argsort(W)
        # W_sorted = W[ind_W]
        # X = torch.as_tensor(batch_states, dtype=torch.float32)
        # U = torch.as_tensor(batch_ctrl, dtype=torch.int32)
        # Weights = torch.as_tensor(batch_weights, dtype=torch.float32)

        X = np.array(batch_states)
        U = np.array(batch_ctrl, dtype=int)
        Weights = np.array(batch_weights)      
        
        # rn = np.arange(W.shape[0], dtype=int)
        # pi = get_policy(X)
        # # prob_pre = np.array([pi.probs[i, batch_ctrl[i]].item() for i in rn])
        # prob_pre = pi.probs[rn, U].detach().numpy()
        # logprob_pre = pi.log_prob(U).detach().numpy()
        # # logits_pre = logits_net(X)[rn, U].detach().numpy()
        # weighted_logp_pre  = logprob_pre  * W
        # ind_logp = np.argsort(weighted_logp_pre)
        # weighted_logp_pre  = weighted_logp_pre[ind_logp]
        
        # take a single policy gradient update step
        batch_loss_pre = compute_loss(x=X, u=U, weights=Weights)
        
        for j in range(len(batch_states)):
            g = compute_loss_grad(batch_states[j], batch_ctrl[j], batch_weights[j])
            logits_x = logits_net(batch_states[j]) 
            logits_x -= lr*g.detach().numpy()

        batch_loss_post = compute_loss(x=X, u=U, weights=Weights)

        print('epoch: %3d \t loss: %.3f \t loss post: %.3f \t delta loss: %.3f\tavg cost per step: %.3f'%
        (i, batch_loss_pre, batch_loss_post, batch_loss_post-batch_loss_pre, np.mean(batch_ctg)/N))
        
        # pi = get_policy(X)
        # # prob_post = np.array([pi.probs[i, batch_ctrl[i]].item() for i in rn])
        # prob_post = pi.probs[rn, U].detach().numpy()
        # logprob_post = pi.log_prob(U).detach().numpy()
        # # logits_post = logits_net(X)[rn, U].detach().numpy()
        # delta_logprob    = logprob_post - logprob_pre
        # delta_prob       = prob_post - prob_pre
        # # delta_logits  = logits_pre[ind_W] - logits_post[ind_W]
        # weighted_logp_post = logprob_post * W
        # weighted_logp_post = weighted_logp_post[ind_logp]

        # plt.figure()
        # plt.plot(W_sorted/np.max(W_sorted), 'o ', label="Normalized weights")
        # plt.plot(delta_prob/np.max(delta_prob), 'x ', label="Delta prob")
        # # plt.plot(delta_logits/np.max(delta_logits), 'x ', label="Delta logits")
        # plt.legend()

        # fig, ax = plut.create_empty_figure(3)
        # ax[0].plot(W[ind_logp], 'o ', label="Weights")
        # ax[0].legend()
        # ax[1].plot(logprob_pre[ind_logp], 'x ', label="Log Prob pre")
        # # ax[1].plot(logprob_post[ind_logp], 'o ', label="Log Prob post")
        # ax[1].plot(delta_logprob[ind_logp], 'o ', label="Delta Log Prob")
        # ax[1].legend()
        # ax[2].plot(prob_pre[ind_logp], 'x ', label="Prob pre")
        # # ax[2].plot(prob_post[ind_logp], 'o ', label="Prob post")
        # ax[2].plot(delta_prob[ind_logp], 'o ', label="Delta Prob")
        # ax[2].legend()

        # fig, ax = plut.create_empty_figure(2)
        # ax[0].plot(weighted_logp_pre,  label="weighted logp pre",  alpha=0.5)
        # ax[0].plot(weighted_logp_post, label="weighted logp post", alpha=0.5)
        # ax[0].plot([0, rn[-1]], 2*[np.mean(weighted_logp_pre)],  ':', label="Avg pre",  alpha=0.5)
        # ax[0].plot([0, rn[-1]], 2*[np.mean(weighted_logp_post)], ':', label="Avg post", alpha=0.5)
        # ax[0].legend()
        # ax[1].plot(weighted_logp_post-weighted_logp_pre, label="delta weighted logp")
        # ax[1].legend()
        # plt.show()

        # for j in range(0, batch_size, 1000):
        #     x = torch.as_tensor(batch_states[j], dtype=torch.float32)
        #     print("P[u(", batch_states[j], ") =", batch_ctrl[j], "]=", get_policy(x).probs[batch_ctrl[j]].item(), ", J=", batch_weights[j])

        # import time
        # time.sleep(1)

        if((i+1)%2==0):
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
        #     # plt.plot(batch_states, (1/N)*np.array(batch_weights), 'x ', label="Cost-to-go", alpha=0.1)
            plt.plot(X_grid, V/(N+1), label="Avg value/step max-lik.", alpha=0.5)

            # plt.figure()
            # plt.plot(X_grid, running_cost, label="cost", alpha=0.9)
            # linestyles = [' x', ' o']
            # for j in range(n_u):
            #     ind = np.where(U==j)[0]
            #     plt.plot(X[ind], Weights[ind], linestyles[j], label="W for u="+str(j), alpha=0.5)
            # plt.legend()

            # average weights for each state bin
            x_discr_init = np.copy(x_min)
            x_discr_end = x_min + x_step
            X_discr, W_discr = np.zeros(N_grid-1), [np.zeros(N_grid-1) for j in range(n_u)]
            for k in range(N_grid-1):
                for j in range(n_u):
                    ind = np.logical_and(U==j, np.logical_and(X>=x_discr_init, X<x_discr_end)).squeeze()
                    if(np.any(ind)):
                        W_discr[j][k] = np.mean(Weights[ind]) / N
                    else:
                        W_discr[j][k] = np.nan 
                X_discr[k] = 0.5*(x_discr_end[0]+x_discr_init[0])
                x_discr_init += x_step
                x_discr_end += x_step
            linestyles = [' x', ' o']
            plt.plot(X_discr, W_discr[0]-W_discr[1], linestyles[1], label="delta W", alpha=0.7)
            # for j in range(n_u):
            #     plt.plot(X_discr, W_discr[j], linestyles[j], label="W for u="+str(j), alpha=0.5)
            plt.legend()

            plt.show()
    