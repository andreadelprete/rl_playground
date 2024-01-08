import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import timeit # start_time = timeit.default_timer()

from rl_playground.utils.function_approximation import NeuralNetwork
from rl_playground.utils.replay_buffer import ReplayBuffer
import rl_playground.utils.plot_utils as plut
from rl_playground.cacto.ocp_single_integrator import OcpSingleIntegrator
from rl_playground.cacto.backward_pass import backward_pass

'''
Test the backward pass to compute the gradient of the Value function
'''
    
w_u = 0.5
dt = 0.1        # time step

@jax.jit
def cost(x_aug, u):
    x = x_aug[0]
    c = w_u*u*u + (x-1.9)*(x-1.0)*(x-0.6)*(x+0.5)*(x+1.2)*(x+2.1)
    return c[0]

@jax.jit
def dynamic(x, u):
    x_next = x + dt*u
    return x_next

@jax.jit
def dynamic_aug(x_aug, u):
    x_next = jnp.array([x_aug[0] + dt*u[0], x_aug[1] + 1])
    return x_next


def plot_critic(t=0, show=True):
    X = replay_buffer.getX()
    ind = jnp.where(X[:,1] == t)[0]
    print("Data points for t=", t, ":", len(ind), "over", X.shape[0])
    X_aug = jnp.vstack([X_grid[:,0], t*np.ones(X_grid.shape[0])]).T

    counter = 1
    plt.figure()
    plt.plot(X_grid, critic(X_aug), label='Critic', alpha=0.5)
    V = replay_buffer.getOut()[ind]
    plt.plot(X[ind,0].T, V, 'x ', label='Value TO')
    plt.xlabel("State x")
    plt.legend()
    plut.saveFigure("Critic_sob_t_"+str(t)+"_v"+str(counter))
    counter += 1
    for critic_sob in critics_sob:
        plt.plot(X_grid, critic_sob(X_aug), label='Critic Sob (ws=%.2f)'%(critic_sob.sobolev_weight), alpha=0.5)
        plt.legend()
        plut.saveFigure("Critic_sob_t_"+str(t)+"_v"+str(counter))
        counter += 1
    if(show): plt.title("Critic for t="+str(t))

    counter = 1
    plt.figure()
    def critic_scalar(x_aug): return critic(x_aug)[0]
    critic_grad = jax.grad(critic_scalar)
    dVdx = jax.vmap(critic_grad)(X_aug)[:,0]
    plt.plot(X_grid, dVdx, label='dVdx critic', alpha=0.5)
    V_x = replay_buffer.getOutGrad()[ind]
    plt.plot(X[ind,0].T, V_x, 'x ', label='dVdx DDP')
    # plt.plot(X_grid, [cost(x) for x in X_grid], alpha=0.5, label="Running cost")
    plt.xlabel("State x")
    plt.legend()
    plut.saveFigure("Critic_grad_sob_t_"+str(t)+"_v"+str(counter))
    counter += 1
    for critic_sob in critics_sob:
        def critic_sob_scalar(x_aug): return critic_sob(x_aug)[0]
        critic_sob_grad = jax.grad(critic_sob_scalar)
        dVdx_sob = jax.vmap(critic_sob_grad)(X_aug)[:,0]
        plt.plot(X_grid, dVdx_sob, label='dVdx critic Sob (ws=%.2f)'%(critic_sob.sobolev_weight), alpha=0.5)
        plt.legend()
        plut.saveFigure("Critic_grad_sob_t_"+str(t)+"_v"+str(counter))
        counter += 1
    # eps = 1e-4
    # X_aug_eps = np.copy(X_aug)
    # X_aug_eps[:,0] += eps
    # plt.plot(X_grid, (critic(X_aug_eps) - critic(X_aug))/eps, 'k:', label='dVdx fin diff')
    if(show):     
        plt.title("Critic gradient for t="+str(t))
        plt.show()


def solve_TO_problem(ocp, N, n_x, n_u, x_init, N_sample):
    sol = ocp.solve(x_init, N_sample)
    J = sol.value(ocp.cost)
    t0 = N-N_sample
    x_aug = jnp.array([sol.value(ocp.x[0]), t0])
    X = np.empty((N_sample+1, n_x))
    U = np.empty((N_sample, n_u))
    for j in range(N_sample):
        X[j,:] = sol.value(ocp.x[j])
        U[j,:] = sol.value(ocp.u[j])
    X[N_sample,:] = sol.value(ocp.x[N_sample])
    V_x = backward_pass(X[:N_sample+1,:], U[:N_sample,:], cost, dynamic, mu=1e-6)
    return (x_aug, J, V_x[0,:])


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    plut.FIGURE_PATH = './critic_sob/'
    N = 10          # horizon size
    n_x, n_u = 1, 1
    x_min, x_max = -2.2, 2.0

    sobolev_weights = [0.01, 0.1, 1]
    learning_rates_critic = [0.0005, 0.0004, 0.0002]
    critic_layers = [8, 8, 8] # number of neurons of the NN layers
    N_OCP = 100
    N_grid = N_OCP
    learning_rate_critic = 0.005
    
    minibatch_size = 128
    critic_updates = 300
    max_critic_iter = 40
    RANDOM_SEED = 0

    ocp = OcpSingleIntegrator(dt, w_u)
    critic     = NeuralNetwork("Value", n_x+1, 1, critic_layers, learning_rate_critic, RANDOM_SEED)
    critics_sob = []
    for (alpha, ws) in zip(learning_rates_critic, sobolev_weights):
        # learning_rate_sob = 0.1*learning_rate_critic / max(1, ws) 
        critics_sob.append(NeuralNetwork("Value", n_x+1, 1, critic_layers, alpha, RANDOM_SEED, ws))
    replay_buffer = ReplayBuffer("V")
    X_grid = jnp.linspace(x_min, x_max, N_grid).reshape((N_grid,n_x))

    print("Start solving TO problems with MP")
    start_time = timeit.default_timer()
    pool = mp.Pool(mp.cpu_count()-1)
    X_samples = np.random.uniform(x_min, x_max, size=(N_OCP, n_x))
    N_samples = np.random.randint(N, N+1, size=N_OCP) # sample in [N, N]
    results = [pool.apply_async(solve_TO_problem, args=(ocp, N, n_x, n_u, X_samples[i,:], N_samples[i])) 
               for i in range(N_OCP)]
    pool.close()   
    for r in results:
        (x, V, Vx) = r.get()
        replay_buffer.append(x, V, Vx)
    print("Finished solving TO problems, which with MP took", timeit.default_timer() - start_time)
    
    X_aug, V_buffer, dVdx_buffer = replay_buffer.getX(), replay_buffer.getOut(), replay_buffer.getOutGrad()
    for critic_sob in critics_sob:
        print("Start training critic with Sobolev and ws=", critic_sob.sobolev_weight)
        for i in range(max_critic_iter):
            critic_loss = critic_sob.train_sobolev(X_aug, V_buffer, dVdx_buffer, 
                                            critic_updates, minibatch_size=minibatch_size)
            print("Iter", i, "\tCritic loss", critic_loss)
        print("Critic training with Sobolev has finished")

    print("Start training critic")
    for i in range(max_critic_iter):
        critic_loss = critic.train(X_aug, V_buffer, critic_updates, minibatch_size=minibatch_size)
        print("Iter", i, "\tCritic loss", critic_loss)
    print("Critic training has finished")

    for t in range(0,N+1,N+1):
        plot_critic(t, show=True)
