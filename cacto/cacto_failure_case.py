import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import timeit # start_time = timeit.default_timer()

from ocp_single_integrator import OcpSingleIntegrator
from rl_playground.utils.function_approximation import NeuralNetwork, ActorNetwork
from rl_playground.utils.replay_buffer import ReplayBuffer
import rl_playground.utils.plot_utils as plut
from backward_pass import backward_pass

'''
Script testing CACTO on a hand-crafted 1d problem where it is supposed to fail learning.
'''

ocp = None

# from single_integrator import *
dt = 0.1        # time step
w_u = 0.15
w_x = 1

@jax.jit
def cost(x, u):
    c = w_u*u*u + w_x*(x-1.9)*(x-1.0)*(x-0.6)*(x+0.5)*(x+1.2)*(x+2.1)
    return c[0]

@jax.jit
def dynamic(x, u):
    x_next = 0.8*x + dt*u
    return x_next
    
@jax.jit
def cost_aug(x_aug, u):
    return cost(x_aug[:-1], u)

@jax.jit
def dynamic_aug(x_aug, u):
    x_next = dynamic(x_aug[:-1], u)
    x_aug_next = jnp.concatenate([x_next, jnp.array([x_aug[-1]+1])])
    return x_aug_next

@jax.jit
def Q_func(u, x_aug):
    return cost_aug(x_aug,u) + critic(dynamic_aug(x_aug,u))[0]


def plot_critic(crit, t=0, show=True):
    X = replay_buffer.getX()
    ind = jnp.where(X[:,1] == t)[0]
    # print("Data points for t=", t, ":", len(ind), "over", X.shape[0])
    X_aug = jnp.vstack([X_grid[:,0], t*np.ones(X_grid.shape[0])]).T

    plt.figure()
    plt.plot(X_grid, crit(X_aug), 'r-', label=crit.name)
    plt.plot(X[ind,0].T, replay_buffer.getOut()[ind], 'x ', label='Value TO')
    plt.plot(X_grid, running_cost, alpha=0.5, label="Cost")
    plt.xlabel("State x")
    # plt.title("Critic for t="+str(t))
    # eps = 1e-4
    # plt.plot(X_grid, (crit(X_grid+eps) - crit(X_grid))/eps, 'k:', label='dVdx')
    if(show): 
        plt.legend()
        if(SHOW_PLOTS): plt.show()

def plot_actor(t=0, show=True, plot_u_ocp=True, plot_running_cost=True):
    X = control_buffer.getX()
    ind = jnp.where(X[:,1] == t)
    X_aug = jnp.vstack([X_grid[:,0], t*np.ones(X_grid.shape[0])]).T
    plt.figure()
    if(plot_u_ocp): plt.plot(X[ind,0].T, control_buffer.getOut()[ind], 'x ', label='u TO')
    if(plot_running_cost): plt.plot(X_grid, running_cost, label="cost", alpha=0.5)
    plt.plot(X_grid, actor(X_aug), 'r-', label='Actor')
    # plt.title("Actor for t="+str(t))
    plt.xlabel("State x")
    if(show): 
        plt.legend()
        if(SHOW_PLOTS): plt.show()

def plot_dQdu(t, plot_u_ocp=True):
    X_aug = jnp.vstack([X_grid[:,0], t*np.ones(X_grid.shape[0])]).T
    grad_Q = jax.grad(Q_func)
    dQdu = jax.vmap(grad_Q)(actor(X_aug), X_aug)
    plt.figure()
    plt.plot(X_grid, critic(X_aug)/N, 'r-', label='Critic')
    plt.plot(X_grid, -dQdu, 'b:', label="-dQdu")
    plt.plot(X_grid, -jnp.sign(dQdu), 'g:', label="-sign(dQdu)")
    plt.plot(X_grid, actor(X_aug), 'k-', label='Actor')
    if(plot_u_ocp): 
        X = control_buffer.getX()
        ind = jnp.where(X[:,1] == t)
        plt.plot(X[ind,0].T, control_buffer.getOut()[ind], 'x ', label='u TO')
    plt.plot()
    plt.title("t="+str(t))
    plt.xlabel("State x")
    plt.legend()
    if(SHOW_PLOTS): plt.show()


def compute_initial_guess(x0, t0, N, n_x, n_u, actor):
    # compute initial guess
    X_guess_aug, U_guess = np.empty((N+1, n_x+1)), np.empty((N, n_u))
    X_guess_aug[0,:] = np.concatenate([x0, np.array([t0])])
    c = 0
    for t in range(N):
        U_guess[t,:] = actor(X_guess_aug[t,:])
        X_guess_aug[t+1,:] = dynamic_aug(X_guess_aug[t,:], U_guess[t,:])
        c += cost_aug(X_guess_aug[t,:], U_guess[t,:])
    c += cost_aug(X_guess_aug[N,:], np.zeros(n_u))
    X_guess = X_guess_aug[:,:-1]
    return X_guess, U_guess, c

def tabular_policy_optimization(U_greedy):
    # Find the minimum of the Q function for plotting the optimally-greedy policy
    print("Start tabular policy optimization")
    for i in range(3000):
        dQdu = jax.vmap(grad_Q)(U_greedy, X_grid_aug)
        U_greedy -= learning_rate_actor_supervised * dQdu
        dQdu_norm = jnp.mean(jnp.abs(dQdu))
        if(i%200==0):
            print("Iter", i, "\tavg(dQdu)=", dQdu_norm)
        if(dQdu_norm < 1e-2):
            print("Tabular policy improvement has converged")
            print("Iter", i, "\tavg(dQdu)=", dQdu_norm)
            break
    print("Tabular policy optimization has finished")
    return U_greedy


def solve_TO_problem(N, n_x, n_u, x_init, N_sample, X_guess=None, U_guess=None):
    ocp = OcpSingleIntegrator(dt, w_u)
    t0 = N-N_sample
    sol = ocp.solve(x_init, N_sample, X_guess, U_guess)
    J = sol.value(ocp.cost)    
    x_aug = jnp.array([sol.value(ocp.x[0]), t0])
    X = np.empty((N_sample+1, n_x))
    U = np.empty((N_sample, n_u))
    for j in range(N_sample):
        X[j,:] = sol.value(ocp.x[j])
        U[j,:] = sol.value(ocp.u[j])
    X[N_sample,:] = sol.value(ocp.x[N_sample])
    V_x = backward_pass(X[:N_sample+1,:], U[:N_sample,:], cost, dynamic, mu=1e-6)
    u = sol.value(ocp.u[0])
    return (x_aug, u, J, V_x[0,:], X, U)


if __name__=="__main__":
    import multiprocessing as mp
    mp.freeze_support()

    N = 10          # horizon size
    n_x, n_u = 1, 1
    x_min, x_max = -2.3, 0.0
    mu_ddp = 1e-6
    SHOW_PLOTS = 1

    critic_layers = [8, 8, 8] # number of neurons of the NN layers
    actor_layers = [8, 8, 8]  # number of neurons of the NN layers
    N_grid = 100
    N_OCP = 2*N_grid # solve one OCP for each state on a grid, for t=0 and t=1
    CACTO_ITERS = 1
    sobolev_weight = 0.1
    learning_rate_critic = 0.002
    learning_rate_actor_supervised = 0.02
    minibatch_size = 128
    critic_updates = 300
    actor_updates = 300
    max_actor_iter = 30
    max_critic_iter = 30
    RANDOM_SEED = 0

    ocp = OcpSingleIntegrator(dt, w_u)
    critic = NeuralNetwork("Critic", n_x+1, 1, critic_layers, learning_rate_critic, RANDOM_SEED, 0.0)
    actor  = ActorNetwork("Policy", n_x+1, n_u, actor_layers, cost_aug, critic, dynamic_aug, 
                        learning_rate_actor_supervised, RANDOM_SEED)
    replay_buffer, control_buffer = ReplayBuffer("V"), ReplayBuffer("U")
    X_grid = jnp.linspace(x_min, x_max, N_grid).reshape((N_grid,n_x))
    X_grid_aug = jnp.vstack([X_grid[:,0], jnp.zeros(X_grid.shape[0])]).T
    X_grid_aug_t1 = jnp.vstack([X_grid[:,0], jnp.ones(X_grid.shape[0])]).T
    running_cost = [cost(X_grid[i,:], 0) for i in range(X_grid.shape[0])]

    # print("Start evaluating OCP's w/o initial guess")
    V = np.empty((N_grid,CACTO_ITERS+1))
    # for (i, x_init) in enumerate(X_grid):
    #     sol = ocp.solve(x_init, N)
    #     V[i,0] = sol.value(ocp.cost)
    # print("Finished solving OCP's")
    # print("Average optimal cost:\n", jnp.mean(V[:,0]))

    k = 0
    print("\n\n\t\t***CACTO - ITERATION", k, "***")

    print("Start solving %d TO problems"%N_OCP)
    replay_buffer.clean()
    control_buffer.clean()
    start_time = timeit.default_timer()
    pool = mp.Pool(mp.cpu_count()-2)
    X_samples = jnp.concatenate([X_grid, X_grid])
    N_samples = np.array([N]*N_grid + [N-1]*N_grid)
    J = np.zeros(N_OCP)
    for i in range(N_OCP):
        (x, u, J[i], Vx, X, U) = solve_TO_problem(N, n_x, n_u, X_samples[i,:], N_samples[i])
        if(i>0 and X_samples[i,0]>-1.0 and np.abs(J[i]-J[i-1]) > 0.1*np.abs(J[i-1])):
            # print("Problem for x0=", X_samples[i,0], "not solved well. J[i]=", J[i], "J[i-1]=", J[i-1])
            (x, u, J[i], Vx, X, U) = solve_TO_problem(N, n_x, n_u, X_samples[i,:], N_samples[i], X_guess, U_guess)
            # print("\t\tNew cost=", J[i])
        X_guess, U_guess = np.copy(X), np.copy(U)
        replay_buffer.append(x, J[i], Vx)
        control_buffer.append(x, u)
        if(N_samples[i]==N):
            V[i,0] = J[i]
    print("Finished solving TO problems, which with MP took", timeit.default_timer() - start_time)

    print("Start training critic")
    X_aug, V_buffer, dVdx_buffer = replay_buffer.getX(), replay_buffer.getOut(), replay_buffer.getOutGrad()
    for i in range(max_critic_iter):
        critic_loss = critic.train(X_aug, V_buffer, critic_updates, minibatch_size=minibatch_size)
        print("Iter", i, "\tCritic loss", critic_loss)
    print("Critic training has finished")
    t = 1
    print("Critic for t=", t)
    plot_critic(critic, t, show=False)
    plt.legend()
    # plut.saveFigure("Iter_"+str(k)+"_critic_t_"+str(t))
    # if(SHOW_PLOTS): plt.show()


    actor_pre_train = actor(X_grid_aug)
    X_aug = replay_buffer.getX()
    grad_Q = jax.jit(jax.grad(Q_func))
    print("Start training actor minimizing Q function")
    U_greedy = jnp.copy(actor_pre_train)
    U_greedy = tabular_policy_optimization(U_greedy)
    
    plt.figure()
    plt.xlabel("State x")
    plt.plot(X_grid, critic(X_grid_aug_t1), label="Critic", alpha=0.5)
    # plot u TO
    X = control_buffer.getX()
    ind = jnp.where(X[:,1] == 0)[0]
    plt.plot(X[ind,0].T, control_buffer.getOut()[ind], 'x ', label='u TO')
    # plot lines connecting each state with next state
    U = actor(X_grid_aug)
    X_next_aug = jax.vmap(dynamic_aug)(X_grid_aug, U)
    V_X = critic(X_grid_aug)
    V_X_next = critic(X_next_aug)
    for i in range(N_grid):
        plt.plot([X_grid[i,0], X_next_aug[i,0]], [U_greedy[i,0], V_X_next[i,0]], alpha=0.2)
    plt.plot(X_grid, U_greedy, ':', label='U greedy pre-training', alpha=0.5)
    plt.legend()
    # if(SHOW_PLOTS): plt.show()

    # plot_dQdu(t=0)

    for i in range(max_actor_iter):
        actor_loss = actor.train_supervised(X_grid_aug, U_greedy, n_iter=actor_updates, 
                                            minibatch_size=minibatch_size, learning_rate=learning_rate_actor_supervised) 
        print("Iter", i, "\tActor loss", actor_loss)
    print("Actor training finished")
    
    # U_greedy_post = actor(X_grid_aug) # initialize search with current policy
    # U_greedy_post = tabular_policy_optimization(U_greedy_post)
    plot_actor(t=0, show=False, plot_u_ocp=True, plot_running_cost=True)
    plt.xlabel("State x")
    plt.plot(X_grid, U_greedy, ':', label='U greedy pre-training', alpha=0.5)
    # plt.plot(X_grid, U_greedy_post, ':', label='U greedy post-training', alpha=0.5)
    plt.legend()
    # if(SHOW_PLOTS): plt.show()

    print("Start evaluating OCP's w actor's initial guess")
    V_guess = np.zeros(N_grid)
    for (i, x_init) in enumerate(X_grid):
        X_guess, U_guess, V_guess[i] = compute_initial_guess(x_init, 0, N, n_x, n_u, actor)
        sol = ocp.solve(x_init, N, X_guess, U_guess)
        V[i,k+1] = sol.value(ocp.cost)
    print("Finished solving OCP's")
    print("Average optimal cost:\n", jnp.mean(V[:,k+1]))

    plt.figure()
    plt.plot(X_grid, V[:,0], ':', label="Value V(x,t=0), iter 0", alpha=0.5)
    for i in range(1, k+2):
        plt.plot(X_grid, V[:,i], label="Value V(x,t=0), iter "+str(i), alpha=0.5)
    # plt.plot(X_grid, V_guess, label="Value current initial guess", alpha=0.5)
    plt.legend()
    plt.xlabel("State x")
    # plut.saveFigure("Iter_"+str(k)+"_to+policy_eval")
    # plt.title("Value functions of OCP with different initial guesses")
    if(SHOW_PLOTS): plt.show()