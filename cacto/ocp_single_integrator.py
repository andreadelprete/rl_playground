import numpy as np
import casadi

'''
A class to solve an Optimal Control Problem (OCP) for a single integrator.
'''
class OcpSingleIntegrator:

    def __init__(self, dt, w_u, w_x=1, u_min=None, u_max=None):
        self.dt = dt
        self.w_u = w_u
        self.w_x = w_x
        self.u_min = u_min
        self.u_max = u_max

    def solve(self, x_init, N, X_guess=None, U_guess=None):
        self.opti = casadi.Opti()
        self.x = self.opti.variable(N+1)
        self.u = self.opti.variable(N)
        x = self.x
        u = self.u

        if(X_guess is not None):
            for i in range(N+1):
                self.opti.set_initial(x[i], X_guess[i,:])
        else:
            for i in range(N+1):
                self.opti.set_initial(x[i], x_init)
        if(U_guess is not None):
            for i in range(N):
                self.opti.set_initial(u[i], U_guess[i,:])

        self.cost = 0
        self.running_costs = [None,]*(N+1)
        for i in range(N+1):
            self.running_costs[i] = self.w_x*(x[i]-1.9)*(x[i]-1.0)*(x[i]-0.6)*(x[i]+0.5)*(x[i]+1.2)*(x[i]+2.1)
            if(i<N):
                self.running_costs[i] += self.w_u * u[i]*u[i]
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        for i in range(N):
            self.opti.subject_to( x[i+1]== 0.8*x[i] + self.dt*u[i] )
        if(self.u_min is not None and self.u_max is not None):
            for i in range(N):
                self.opti.subject_to( self.opti.bounded(self.u_min, u[i], self.u_max) )
        self.opti.subject_to(x[0]==x_init)

        # s_opts = {"max_iter": 100}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts) #, s_opts)

        return self.opti.solve()


if __name__=="__main__":
    import matplotlib.pyplot as plt
    N = 10          # horizon size
    dt = 0.1        # time step
    x_init = -0.8   # initial state
    w_u = 1e-2
    u_min = -1      # min control input
    u_max = 1       # max control input


    ocp = OcpSingleIntegrator(dt, w_u, u_min, u_max)
    sol = ocp.solve(x_init, N)
    print("Optimal value of x:\n", sol.value(ocp.x))

    X = np.linspace(-2.2, 2.0, 100)
    costs = [sol.value(ocp.running_costs[0], [ocp.x==x_val]) for x_val in X]
    # costs = [sol.value(running_costs[0], [x[0]==x_val]) for x_val in X]
    plt.plot(X, costs)
    for i in range(N+1):
        plt.plot(sol.value(ocp.x[i]), sol.value(ocp.running_costs[i]), 
                'xr', label='x_'+str(i))
    plt.legend()
    plt.show()
