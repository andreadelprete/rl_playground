import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import timeit # start_time = timeit.default_timer()

from rl_playground.utils.function_approximation import NeuralNetwork, ActorNetwork
from rl_playground.utils.replay_buffer import ReplayBuffer
import rl_playground.utils.plot_utils as plut
from rl_playground.cacto.ocp_single_integrator import OcpSingleIntegrator
from rl_playground.cacto.backward_pass import backward_pass, backward_pass_box

'''
Test the box backward pass to compute the gradient of the Value function
for a TO problem with control bounds.
'''
    
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

    plt.figure()
    plt.plot(X_grid, critic(X_aug), 'r-', label='Critic')
    V = replay_buffer.getOut()[ind]
    plt.plot(X[ind,0].T, V, 'x ', label='Value TO')
    V_x = replay_buffer.getOutGrad()[ind]
    plt.plot(X[ind,0].T, V_x, 'o ', label='dVdx DDP')
    # plt.plot(X_grid, running_cost, alpha=0.5, label="Running cost")
    plt.xlabel("State x")
    plt.title("Critic for t="+str(t))
    eps = 1e-4
    X_aug_eps = np.copy(X_aug)
    X_aug_eps[:,0] += eps
    plt.plot(X_grid, (critic(X_aug_eps) - critic(X_aug))/eps, 'k:', label='dVdx finite-diff critic')

    X = replay_buffer_fd.getX()
    ind = jnp.where(X[:,1] == t)[0]
    V_x_fd = replay_buffer_fd.getOutGrad()[ind]
    plt.plot(X[ind,0].T, V_x_fd, 'v ', label='dVdx finite-diff TO')
    if(show): 
        plt.legend()
        plt.show()

N = 10          # horizon size
dt = 0.1        # time step
n_x, n_u = 1, 1
w_u = 0.1
x_min, x_max = -2.2, 2.0
u_min, u_max = -2.0, 2.0

critic_layers = [8, 8, 8] # number of neurons of the NN layers
N_OCP = 50
N_grid = N_OCP
CACTO_ITERS = 2
learning_rate_critic = 0.005
minibatch_size = 128
critic_updates = 300
critic_loss_thr = 1e-3
max_critic_iter = 10
RANDOM_SEED = 0
mu = 1e-6 # DDP backward pass regularization

ocp = OcpSingleIntegrator(dt, w_u, u_min=u_min, u_max=u_max)
critic = NeuralNetwork("Value", n_x+1, 1, critic_layers, learning_rate_critic, RANDOM_SEED)
replay_buffer = ReplayBuffer("V")
replay_buffer_fd = ReplayBuffer("V fd")
X_grid = jnp.linspace(x_min, x_max, N_grid).reshape((N_grid,n_x))

print("Start solving TO problems")
V = np.empty((N_grid, N+1))
V_x = np.empty((N_grid, N+1, n_x))
X = np.empty((N+1, n_x))
U = np.empty((N, n_u))
eps = 1e-5
for i in range(N_OCP):
    x_sample = X_grid[i%N_grid,:]
    N_sample = np.random.randint(N, N+1) # sample in [1, N]
    sol = ocp.solve(x_sample, N_sample)

    J = sol.value(ocp.cost)
    t0 = N-N_sample
    x_aug = jnp.array([sol.value(ocp.x[0]), t0])
    for j in range(N_sample):
        X[j,:] = sol.value(ocp.x[j])
        U[j,:] = sol.value(ocp.u[j])
    X[N_sample,:] = sol.value(ocp.x[N_sample])
    # V_x = backward_pass(X[:N_sample+1,:], U[:N_sample,:], cost, dynamic, mu)
    V_x = backward_pass_box(X[:N_sample+1,:], U[:N_sample,:], cost, dynamic, 
                            mu, u_min, u_max)
    replay_buffer.append(x_aug, J, V_x[0,:])

    x_sample_eps = x_sample.at[0].add(eps)
    sol_eps = ocp.solve(x_sample_eps, N_sample)
    J_eps = sol_eps.value(ocp.cost)
    replay_buffer_fd.append(x_aug, J, np.array([(J_eps-J)/eps]))


running_cost = [sol_eps.value(ocp.running_costs[0], [ocp.x==x_val]) for x_val in X_grid]
print("Finished solving TO problems")

print("Start training critic")
X_aug, V_buffer = replay_buffer.getX(), replay_buffer.getOut()
for i in range(max_critic_iter):
    critic_loss = critic.train(X_aug, V_buffer, critic_updates, minibatch_size=minibatch_size)
    print("Iter", i, "\tCritic loss", critic_loss)
    if(i%10==11):
        plot_critic(t=0, show=True)
print("Critic training has finished")

for t in range(0,N+1,N+1):
    plot_critic(t, show=True)
