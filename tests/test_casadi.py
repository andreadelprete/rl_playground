import numpy as np
import casadi
import matplotlib.pyplot as plt

opti = casadi.Opti()

x = opti.variable()
# y = opti.variable()
cost = (x-1.9)*(x-1.0)*(x-0.6)*(x+0.5)*(x+1.2)*(x+2.1)
opti.minimize(  cost )
# opti.subject_to( x**2+y**2==1 )
# opti.subject_to(       x+y>=1 )

opti.solver('ipopt')

opti.set_initial(x, -2)
# opti.set_initial(10*x[0], 2)
# opti.set_initial(sol.value_variables())

# To initialize dual variables:
# lam_g0 = sol.value(opti.lam_g)
# opti.set_initial(opti.lam_g, lam_g0)

sol = opti.solve()

print(sol.value(x))

X = np.linspace(-2.5, 2.5, 100)
costs = [sol.value(cost, [x==x_val]) for x_val in X]
plt.plot(X, costs)
plt.plot(sol.value(x), sol.value(cost), 'xr', label='optimal solution')
plt.legend()
plt.show()
