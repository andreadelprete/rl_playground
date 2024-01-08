import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import timeit # start_time = timeit.default_timer()

from ocp_single_integrator import OcpSingleIntegrator
from rl_playground.utils.function_approximation import NeuralNetwork, ActorNetwork
from rl_playground.utils.replay_buffer import ReplayBuffer
import rl_playground.utils.plot_utils as plut

'''
Simple script testing the first iteration of CACTO on a 1d single integrator.
'''

from rl_playground.utils.single_integrator import *

@jax.jit
def Q_func(u, x_aug):
    return cost(x_aug,u) + critic(dynamic(x_aug,u))[0]


def plot_critic(t=0, show=True):
    X = replay_buffer.getX()
    ind = jnp.where(X[:,1] == t)[0]
    print("Data points for t=", t, ":", len(ind), "over", X.shape[0])
    X_aug = jnp.vstack([X_grid[:,0], t*np.ones(X_grid.shape[0])]).T

    plt.figure()
    plt.plot(X[ind,0].T, replay_buffer.getOut()[ind], 'x ', label='Value')
    plt.plot(X_grid, critic(X_aug), 'r-', label='Critic')
    plt.title("Critic for t="+str(t))
    # eps = 1e-4
    # plt.plot(X_grid, (critic(X_grid+eps) - critic(X_grid))/eps, 'k:', label='dVdx')
    plt.legend()
    if(show): plt.show()

def plot_actor(t=0, show=True, plot_u_ocp=True, plot_running_cost=True):
    X = control_buffer.getX()
    ind = jnp.where(X[:,1] == t)
    X_aug = jnp.vstack([X_grid[:,0], t*np.ones(X_grid.shape[0])]).T
    plt.figure()
    if(plot_u_ocp): plt.plot(X[ind,0].T, control_buffer.getOut()[ind], 'x ', label='u OCP')
    if(plot_running_cost): plt.plot(X_grid, running_cost, label="running cost")
    plt.plot(X_grid, actor(X_aug), 'b-', label='Actor')
    plt.title("Actor for t="+str(t))
    if(show): 
        plt.legend()
        plt.show()

def plot_dQdu(t):
    X_aug = jnp.vstack([X_grid[:,0], t*np.ones(X_grid.shape[0])]).T
    grad_Q = jax.grad(Q_func)
    dQdu = np.array([grad_Q(actor(x_aug), x_aug)[0] for x_aug in X_aug])
    plt.figure()
    plt.plot(X_grid, 0.2*critic(X_aug), 'r-', label='Critic')
    plt.plot(X_grid, -dQdu, 'b:', label="-dQdu")
    # plt.plot(X_grid, -dQdu_fd, 'g:', label="-dQdu FD")
    plt.plot(X_grid, -np.sign(dQdu), 'g:', label="-sign(dQdu)")
    plt.plot(X_grid, actor(X_aug), 'k-', label='Actor')
    # plt.plot(X_grid, u_ocp, 'xk ', label='U OCP')
    plt.title("t="+str(t))
    plt.legend()
    plt.show()


N = 10          # horizon size
dt = 0.1        # time step
n_x, n_u = 1, 1
x_min, x_max = -2.2, 2.0
w_u = 1e-1

critic_layers = [8, 8, 8] # number of neurons of the NN layers
actor_layers = [8, 8, 8]  # number of neurons of the NN layers
N_x = 1000
learning_rate_critic = 0.005
learning_rate_actor = 0.01
minibatch_size = 64
critic_updates = 300
actor_updates = 300
critic_loss_thr = 2.0
actor_loss_thr = 0.1
actor_grad_thr = 0.6
RANDOM_SEED = 0


ocp = OcpSingleIntegrator(dt, w_u)
X_grid = jnp.linspace(x_min, x_max, N_x).reshape((N_x,n_x))
replay_buffer, control_buffer = ReplayBuffer("V"), ReplayBuffer("U")
print("Start solving OCP's")
for i in range(N_x):
    # x_sample = np.random.uniform(x_min, x_max)
    x_sample = X_grid[i,:]
    N_sample = np.random.randint(1, N+1) # sample in [1, N]
    sol = ocp.solve(x_sample, N_sample)
    J = sol.value(ocp.cost)
    x_aug = jnp.array([sol.value(ocp.x[0]), N-N_sample])
    replay_buffer.append(x_aug, J)
    u = sol.value(ocp.u[0])
    control_buffer.append(x_aug, u)
    # for t in range(N_sample, -1, -1):
    #     J += sol.value(ocp.running_costs[t])
    #     x_aug = jnp.array([sol.value(ocp.x[t]), t+N-N_sample])
    #     replay_buffer.append(x_aug, J)
    #     if(t<N_sample):
    #         u = sol.value(ocp.u[t])
    #         control_buffer.append(x_aug, u)
print("Finished solving OCP's")
running_cost = [sol.value(ocp.running_costs[0], [ocp.x==x_val]) for x_val in X_grid]

critic = NeuralNetwork("Value", n_x+1, 1, critic_layers, learning_rate_critic, RANDOM_SEED)
X_aug, V_buffer = replay_buffer.getX(), replay_buffer.getOut()
for i in range(50):
    critic_loss = critic.train(X_aug, V_buffer, critic_updates, minibatch_size=minibatch_size)
    print("Iter", i, "\n\tCritic loss", critic_loss)
    if(critic_loss < critic_loss_thr):
        print("Critic loss has converged")
        break
for t in range(1, N, N):
    print("Critic for t=", t)
    plot_critic(t)

actor  = ActorNetwork("Policy", n_x+1, n_u, actor_layers, cost, critic, dynamic, 
                      learning_rate_actor, RANDOM_SEED)

print("Start training actor with supervised learning")
X_aug, U_buffer = control_buffer.getX(), control_buffer.getOut()
for i in range(50):
    actor_loss = actor.train_supervised(X_aug, U_buffer, actor_updates, minibatch_size=minibatch_size)
    print("Iter", i, "\n\tActor supervised loss", actor_loss)
    if(actor_loss < actor_loss_thr):
        print("Actor loss has converged")
        break
for t in range(0, N, N):
    print("Actor for t=", t)
    plot_actor(t)

print("Start training actor minimizing Q function")
# plot_dQdu(t=0)
X_grid_aug = jnp.vstack([X_grid[:,0], 0*np.ones(X_grid.shape[0])]).T
actor_pre_train = actor(X_grid_aug)
grad_Q = jax.jit(jax.grad(Q_func))
X_aug = replay_buffer.getX()

for i in range(50):
    actor_loss = actor.train(X_aug, actor_updates, minibatch_size=minibatch_size) 
    
    # start_time = timeit.default_timer()
    U = actor(X_aug)
    # dQdu = jnp.mean(jnp.abs(jax.vmap(grad_Q)(U, X_aug))).block_until_ready()
    dQdu = jnp.mean(jnp.abs(jax.vmap(grad_Q)(U, X_aug)))
    # print("Time dQdu with vmap", timeit.default_timer()-start_time)
    
    # start_time = timeit.default_timer()
    # dQdu2 = np.array([grad_Q(actor(x_aug), x_aug)[0] for x_aug in X_aug])
    # print("Time dQdu          ", timeit.default_timer()-start_time)

    print("Iter", i, "\n\tActor loss", actor_loss, "\n\tdQdu", dQdu)
    if(dQdu < actor_grad_thr):
        print("Actor loss grad norm has converged")
        break
    if(i%10==0):
        plot_actor(t=0, show=False)
        plt.plot(X_grid, actor_pre_train, 'r-', label='Actor pre-training')
        plt.legend()
        plt.show()

print("Actor training finished")

plot_actor(t=0, show=False)
plt.plot(X_grid, actor_pre_train, 'r-', label='Actor pre-training')
plt.legend()
plt.show()

plot_dQdu(t=0)

# Find the minimum of the Q function for plotting the optimally-greedy policy
print("Start tabular policy optimization")
U_greedy = actor(X_grid_aug) # initialize search with current policy
for i in range(100):
    dQdu = jax.vmap(grad_Q)(U_greedy, X_grid_aug)
    U_greedy -= learning_rate_actor * dQdu
    dQdu_norm = jnp.mean(jnp.abs(dQdu))
    if(i%10==0):
        print("Iter", i)
        print("\tavg(dQdu)=", dQdu_norm)
    if(dQdu_norm < 1e-2):
        print("Tabular policy improvement has converged")
        break

plot_actor(t=0, show=False, plot_u_ocp=False, plot_running_cost=False)
plt.plot(X_grid, U_greedy, 'r-', label='Tabular greedy policy')
plt.plot(X_grid, U_greedy-actor(X_grid_aug), 'b--', label='U_greedy - U_actor')
U = actor(X_grid_aug)
dQdu = jax.vmap(grad_Q)(U, X_grid_aug)
plt.plot(X_grid, -dQdu, "g:", label="-dQdu actor")
plt.legend()
plt.show()