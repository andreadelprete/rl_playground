import jax
import jax.numpy as jnp

dt = 0.1        # time step
w_u = 0.1

# @jax.jit
def cost(x, u):
    x = x[0]
    c = w_u*u*u + (x-1.9)*(x-1.0)*(x-0.6)*(x+0.5)*(x+1.2)*(x+2.1)
    return c[0]
    
# @jax.jit
def cost_aug(x_aug, u):
    return cost(x_aug[:-1], u)

# @jax.jit
def dynamic(x, u):
    x_next = x + dt*u
    return x_next

# @jax.jit
def dynamic_aug(x_aug, u):
    x_next = dynamic(x_aug[:-1], u)
    x_aug_next = jnp.concatenate([x_next, jnp.array([x_aug[-1]+1])])
    return x_aug_next
