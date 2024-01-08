from rl_playground.cacto.ocp_single_integrator import OcpSingleIntegrator
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import rl_playground.utils.plot_utils as plut

'''
Simple script testing the warm start of Casadi
'''

@jax.jit
def cost(x, u):
    c = w_u*u*u + (x-1.9)*(x-1.0)*(x-0.6)*(x+0.5)*(x+1.2)*(x+2.1)
    return c[0]

@jax.jit
def dynamic(x, u):
    x_next = x + dt*u
    return x_next

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


N = 10          # horizon size
w_u = 0.5
dt = 0.1        # time step
n_x, n_u = 1, 1
# u_min, u_max = -1, 1      # min/max control input
x_min, x_max = -2.2, 2.0
N_x = 200

ocp = OcpSingleIntegrator(dt, w_u)
X_init = np.linspace(x_min, x_max, N_x).reshape((N_x,n_x))
V = np.empty((N_x,1))
V_guess = np.empty((N_x,1))
V_no_ws = np.empty((N_x,1))
u_ocp = np.empty((N_x,n_u))
u_guess = np.empty((N_x,n_u))
print("Start solving OCP's")
U_guess = -np.ones((N, n_u))
X_guess = np.empty((N+1, n_x))
for (i, x_init) in enumerate(X_init):
    sol_no_ws = ocp.solve(x_init, N)
    V_no_ws[i] = sol_no_ws.value(ocp.cost)
    u_ocp[i] = sol_no_ws.value(ocp.u[0])
    
    X_guess, U_guess, V_guess[i] = compute_initial_guess(x_init, N, n_x, n_u)
    sol = ocp.solve(x_init, N, X_guess, U_guess)
    V[i] = sol.value(ocp.cost)
    # u_ocp[i] = sol.value(ocp.u[0])
    u_guess[i] = U_guess[0,0]
print("Finished solving OCP's")

# running_cost = [sol.value(ocp.running_costs[0], [ocp.x==x_val]) for x_val in X_init]
running_cost = [cost(x, 0) for x in X_init]
plt.plot(X_init, running_cost, label="Cost c(x)")
# plt.plot(X_init, u_ocp, 'x ', label='Control u(x,t=0)')
# plt.plot(X_init, u_guess, 'x ', label='Control Guess')
# plt.plot(X_init, V, 'x ', label='Value OCP w/ warm start')
# plt.plot(X_init, V_guess, 'x ', label='Value initial guess')
# plt.plot(X_init, V_no_ws, 'x ', label='Value V(x,t=0)')
plt.xlabel("State x")
plt.legend()
plt.show()
