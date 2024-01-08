import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from ocp_single_integrator import OcpSingleIntegrator
from rl_playground.utils.function_approximation import NeuralNetwork, ActorNetwork
import rl_playground.utils.plot_utils as plut

'''
Simple script testing the basics of CACTO on a 1d single integrator.
Solve OCP's, learn the Value function using the OCP data, and then learn a policy
minimizing the Q function associated to the learned Value function.
Works exclusively on the first time step to simplify things.
'''

@jax.jit
def cost(x, u):
    c = w_u*u*u + w_x*(x-1.9)*(x-1.0)*(x-0.6)*(x+0.5)*(x+1.2)*(x+2.1)
    return c[0]

@jax.jit
def dynamic(x, u):
    x_next = 0.8*x + dt*u
    return x_next

@jax.jit
def Q_func(u, x):
    return cost(x,u) + critic(dynamic(x,u))[0]

def compute_initial_guess(x0, N, n_x, n_u):
    # compute initial guess
    X_guess, U_guess = np.empty((N+1, n_x)), np.empty((N, n_u))
    X_guess[0,0] = x0
    kp = 6
    c = 0
    for t in range(N):
        U_guess[t,:] =  kp * (-1.8 - X_guess[t,0])
        X_guess[t+1,:] = dynamic(X_guess[t,:], U_guess[t,:])
        c += cost(X_guess[t,:], U_guess[t,:])
    c += cost(X_guess[N,:], 0)
    return X_guess, U_guess, c


def plot_critic(show=True):
    eps = 1e-4
    plt.figure()
    plt.plot(X_init, V, 'x ', label='Value')
    plt.plot(X_init, critic(X_init), 'r-', label='Critic')
    # plt.plot(X_init, (critic(X_init+eps) - critic(X_init))/eps, 'k:', label='dVdx')
    plt.legend()
    if(show): plt.show()

def plot_actor(show=True):
    plt.figure()
    plt.plot(X_init, u_ocp, 'x ', label='u OCP')
    plt.plot(X_init, running_cost, label="running cost")
    plt.plot(X_init, actor(X_init), 'r-', label='Actor')
    plt.legend()
    if(show): plt.show()

def plot_Q():
    N_u = 21
    U_grid = np.linspace(u_min, u_max, N_u).reshape((N_u,n_u))
    X,U = np.meshgrid(range(N_x),range(N_u))
    Q = np.zeros((N_x, N_u))
    for i in range(N_x):
        for j in range(N_u):
            Q[i,j] = cost(X_init[i], U_grid[j]) + critic(dynamic(X_init[i], U_grid[j]))
    plt.pcolormesh(X, U, Q.T, cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar()
    plt.title('Q function')
    plt.xlabel("x")
    plt.ylabel("u")
    plt.show()

def plot_Q_actor():
    plt.figure()
    plt.plot(X_init, critic(X_init), 'r-', label='Critic')
    Q = [cost(x, actor(x)) + critic(dynamic(x, actor(x))) for x in X_init]
    plt.plot(X_init, Q, 'b:', label="Q actor")
    plt.plot(X_init, actor(X_init), 'k-', label='Actor')
    plt.legend()
    plt.show()

def plot_dQdu():
    grad_Q = jax.grad(Q_func)
    dQdu = np.array([grad_Q(actor(x), x)[0] for x in X_init])

    # eps = 1e-4
    # dQdu_fd = np.zeros(N_x)
    # Q     = np.array([cost(x, actor(x))     + critic(dynamic(x, actor(x)))     for x in X_init])
    # Q_eps = np.array([cost(x, actor(x)+eps) + critic(dynamic(x, actor(x)+eps)) for x in X_init])
    # dQdu_fd = (Q_eps - Q)/eps

    plt.figure()
    # plt.plot(X_init, 0.2*critic(X_init), 'r-', label='Critic')
    plt.plot(X_init, -dQdu, 'b:', label="-dQdu")
    # plt.plot(X_init, -dQdu_fd, 'g:', label="-dQdu FD")
    plt.plot(X_init, -np.sign(dQdu), 'g:', label="-sign(dQdu)")
    plt.plot(X_init, actor(X_init), 'k-', label='Actor')
    plt.plot(X_init, u_ocp, 'xk ', label='U OCP')
    plt.legend()
    plt.show()


N = 10          # horizon size
dt = 0.1        # time step
n_x, n_u = 1, 1
w_u = 0.15
w_x = 1
x_min, x_max = -2.2, 2.0
u_min, u_max = -1, 1

critic_layers = [8, 8, 8] # number of neurons of the NN layers
actor_layers = [8, 8, 8]  # number of neurons of the NN layers
N_x = 200
learning_rate_critic = 0.003
learning_rate_actor = 0.01
minibatch_size = 64
critic_updates = 300
actor_updates = 300
critic_iters = 30
actor_loss_thr = 0.2
actor_grad_thr = 5.0
RANDOM_SEED = 0

ocp = OcpSingleIntegrator(dt, w_u, w_x)
X_init = np.linspace(x_min, x_max, N_x).reshape((N_x,n_x))
V = np.empty((N_x,1))
u_ocp = np.empty((N_x,n_u))
print("Start solving OCP's")
for (i, x_init) in enumerate(X_init):
    sol = ocp.solve(x_init, N)
    V[i] = sol.value(ocp.cost)
    u_ocp[i] = sol.value(ocp.u[0])
    # print("Optimal cost:\n", V[i])
print("Finished solving OCP's")

running_cost = [sol.value(ocp.running_costs[0], [ocp.x==x_val]) for x_val in X_init]
plt.plot(X_init, running_cost, label="cost c(x)")
plt.plot(X_init, u_ocp, 'x ', label='OCP Control')
plt.plot(X_init, V/(N+1), 'x ', label='Value OCP')
plt.legend()
plt.show()

critic = NeuralNetwork("Value", n_x, 1, critic_layers, learning_rate_critic, RANDOM_SEED)
for i in range(critic_iters):
    critic_loss = critic.train(X_init, V, critic_updates, minibatch_size=minibatch_size)
    print("Iter", i, "\tCritic loss", critic_loss)
plot_critic()

actor  = ActorNetwork("Policy", n_x, n_u, actor_layers, cost, critic, dynamic, 
                      learning_rate_actor, RANDOM_SEED)
plot_dQdu()

print("Start training actor with supervised learning")
plot_actor()
for i in range(100):
    actor_loss = actor.train_supervised(X_init, u_ocp, actor_updates, minibatch_size=minibatch_size)
    print("Iter", i, "\tActor supervised loss", actor_loss)
    if(actor_loss < actor_loss_thr):
        print("Actor loss has converged")
        break
plot_actor()

print("Start training actor minimizing Q function")
plt.figure()
plt.plot(X_init, u_ocp, 'x ', label='u OCP')
plt.plot(X_init, running_cost, label="running cost")
plt.plot(X_init, actor(X_init), 'r-', label='Actor pre-training')

grad_Q = jax.grad(Q_func)
for i in range(100):
    actor_loss = actor.train(X_init, actor_updates, minibatch_size=minibatch_size)
    actor_loss, actor_grads = actor.loss_grad_fn(actor.params, X_init)
    grad_norm = 0
    for key in actor_grads['params'].keys():
        grad_norm += np.linalg.norm(actor_grads['params'][key]['bias'])
        grad_norm += np.linalg.norm(actor_grads['params'][key]['kernel'])
    
    dQdu = np.array([grad_Q(actor(x), x)[0] for x in X_init])
    print("Iter", i, "\n\tActor loss", actor_loss, "\n\tActor grad", grad_norm, 
          "\n\tdQdu", np.linalg.norm(dQdu))
    if(grad_norm < actor_grad_thr):
        print("Actor loss grad norm has converged")
        break
    # plot_actor()

plt.plot(X_init, actor(X_init), 'b:', label='Actor post-training')
plt.legend()
plt.show()

plot_dQdu()
