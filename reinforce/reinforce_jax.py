import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax.random import normal
import matplotlib.pyplot as plt

from rl_playground.utils.function_approximation import NeuralNetwork, ActorNetwork, update_params
from rl_playground.utils.single_integrator import *
import rl_playground.utils.plot_utils as plut

''' Basic implementation of Reinforce algorithm with continuous action space, 
    i.e. policy gradient using empirical return as weight for each trajectory.
    Implementation based on Jax for representing the policy as a NN, and
    tested on a simple non-convex 1D single integrator problem.

    NOT WORKING!
'''
def sample_action(key, actor, x):
    mu_sigma = actor(x)
    mu, sigma = mu_sigma[:n_u], mu_sigma[n_u:]
    k1, k2 = random.split(key)
    u = mu + sigma * normal(k1, shape=(n_u,))
    return u, k2

def log_pi(params, u, x, actor):
    mu_sigma = actor.model.apply(params, x)
    mu, sigma = mu_sigma[:n_u], jnp.abs(mu_sigma[n_u:])
    res = 0
    for i in range(n_u):
        res += ((u[i] - mu[i])**2 / sigma[i]**2) + 2*jnp.log(sigma[i])
    res = -0.5*(res + n_u*jnp.log(2*jnp.pi))
    return res

def plot_actor(show=True):
    plt.figure()
    plt.plot(X_grid, running_cost, label="cost", alpha=0.5)
    mu_sigma = actor(X_grid)
    mu, sigma = mu_sigma[:,0], jnp.abs(mu_sigma[:,1])
    plt.plot(X_grid, mu, 'r', label='Actor')
    plt.plot(X_grid, mu+sigma, 'r:', label='Actor+sigma')
    plt.plot(X_grid, mu-sigma, 'r:', label='Actor-sigma')
    plt.title("Actor")
    plt.xlabel("State x")
    if(show): 
        plt.legend()
        plt.show()

if __name__=="__main__":
    import multiprocessing as mp
    mp.freeze_support()

    N = 10          # horizon size
    n_x, n_u = 1, 1
    x_min, x_max = jnp.array([-2.2]), jnp.array([2.0])
    u_min, u_max = jnp.array([-2.0]), jnp.array([2.0])
    SHOW_PLOTS = 1
    actor_layers = [8, 8, 8]  # number of neurons of the NN layers
    learning_rate_actor = 1e-5
    minibatch_size = 128
    actor_updates = 300
    max_actor_iter = 35
    RANDOM_SEED = 0
    N_EPISODES = 100 # => batch_size=N_EPISODES*N=5000
    N_EPOCHS = 10
    N_grid = 10
    

    key = random.key(RANDOM_SEED)
    actor = ActorNetwork("Policy", n_x, 2*n_u, actor_layers, cost, None, dynamic, 
                        learning_rate_actor, RANDOM_SEED)
    X_grid = jnp.linspace(x_min, x_max, N_grid).reshape((N_grid,n_x))
    # U_grid = jnp.linspace(u_min, u_max, N_grid).reshape((N_grid,n_u))
    running_cost = [cost(np.array([x]), 0) for x in X_grid]
    
    x = np.random.uniform(x_min, x_max, size=(n_x))
    plot_actor(show=False)
    plt.plot(x, actor(x)[0], 'b o', label="Chosen state")
    plt.legend()

    mu_sigma = actor(x)
    mu, sigma = mu_sigma[0], jnp.abs(mu_sigma[1])
    U_grid = jnp.linspace(mu-2*sigma, mu+2*sigma, N_grid).reshape((N_grid,n_u))
    lp_grid = [log_pi(actor.params, u, x, actor) for u in U_grid]
    print(lp_grid)
    plt.figure()
    plt.plot(U_grid, lp_grid, label="log prob of u")
    plt.plot(mu, np.zeros(1), 'x ', label="mu")
    plt.plot(mu-sigma, np.zeros(1), 'o ', label="mu-sigma")
    plt.plot(mu+sigma, np.zeros(1), 'o ', label="mu+sigma")
    plt.legend()
    plt.show()

    for epoch in range(N_EPOCHS):
        print("Epoch", epoch)
        # make some empty lists for logging.
        batch_states = []       # for states
        batch_controls = []     # for controls
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_cost_to_go = []   # for measuring episode cost-to-go's
        batch_lens = []         # for measuring episode lengths

        # collect experience by acting in the environment with current policy
        for ep in range(N_EPISODES):
            # reset episode-specific variables
            ep_costs = []            # list for costs accrued throughout ep
            x = np.random.uniform(x_min, x_max, size=(n_x))
            for t in range(N):
                u, key = sample_action(key, actor, x)
                c = cost(x, u)
                x = dynamic(x, u)
                x = jnp.maximum(x_min, np.minimum(x, x_max))
                # save obs, action, reward
                batch_states.append(x.copy())
                batch_controls.append(u)
                ep_costs.append(c)

            # when episode is over, record info about episode
            ep_ret = sum(ep_costs)
            batch_cost_to_go.append(ep_ret)
            # batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            batch_weights += [ep_ret] * N

            # print("Episode", ep, "Return", ep_ret)

            # plt.figure()
            # plt.plot(X_grid, running_cost, label="running cost")
            # plt.plot(batch_states[-N:], batch_controls[-N:], 'o ', label="episode controls")
            # plt.plot(batch_states[-N:], ep_costs, 'x ', label="episode costs")
            # plt.legend()
            # plt.show()
        print("Epoch", epoch, "ended with avg return", jnp.mean(jnp.array(batch_weights)))

        # make loss function whose gradient, for the right data, is policy gradient
        def compute_loss(params, x_batched, u_batched, weights_batched):
            def single_loss(x, u, w):
                logp = log_pi(params, u, x, actor)
                return jnp.mean(logp * w)
            # Vectorize the previous to compute the average of the loss on all samples.
            return jnp.mean(jax.vmap(single_loss)(x_batched, u_batched, weights_batched), axis=0)

        # take a single policy gradient update step            
        loss_grad = jax.value_and_grad(compute_loss)
        # Perform one gradient update.    
        loss_val, grads = loss_grad(actor.params, jnp.array(batch_states), 
                                    jnp.array(batch_controls), jnp.array(batch_weights))
        actor.params = update_params(actor.params, actor.learning_rate, grads)
        plot_actor()
