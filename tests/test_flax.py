from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from jax import random
import matplotlib.pyplot as plt


class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.elu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x

# Set problem dimensions.
RANDOM_KEY = 3
n_samples = 100
x_dim = 1
y_dim = 1
learning_rate = 0.1  # Gradient step size.
n_iter = 1000
n_print = 100
function_type = 'sin' # linear or sin_

model = MLP([32, 32, y_dim])
k1, k2 = random.split(random.key(RANDOM_KEY))
x = random.normal(k1, (x_dim,)) # Dummy input data
params = model.init(k2, x) # Initialization call
jax.tree_util.tree_map(lambda x: x.shape, params) # Checking output shapes

# Generate samples with additional noise.
key_sample, key_noise = random.split(k1)
x_samples = random.uniform(key_sample, (n_samples, x_dim))
# Generate linear data with random W and b.
if(function_type=='linear'):
  W = random.normal(k1, (x_dim, y_dim))
  b = random.normal(k2, (y_dim,))
  y_samples = jnp.dot(x_samples, W) + b + 0.001 * random.normal(key_noise,(n_samples, y_dim))
elif(function_type=='sin'):
  y_samples = jnp.sin(3*x_samples)
# print('x shape:', x_samples.shape, '; y shape:', y_samples.shape)

# Same as JAX version but using model.apply().
@jax.jit
def mse(params, x_batched, y_batched):
  # Define the squared loss for a single pair (x,y)
  def squared_error(x, y):
    pred = model.apply(params, x)
    return jnp.inner(y-pred, y-pred) / 2.0
  # Vectorize the previous to compute the average of the loss on all samples.
  return jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0)

print('Loss for initial network weights: ', mse(params, x_samples, y_samples))
loss_grad_fn = jax.value_and_grad(mse)

@jax.jit
def update_params(params, learning_rate, grads):
  params = jax.tree_util.tree_map(
      lambda p, g: p - learning_rate * g, params, grads)
  return params

def plot_predictions(x_samples, y_samples, model, params):
  plt.plot(x_samples, y_samples, 'xr ', label='Training data')
  y_pred = model.apply(params, x_samples)
  plt.plot(x_samples, y_pred, 'xk ', label='Prediction')
  plt.legend()
  plt.show()
  
for i in range(n_iter):
  # Perform one gradient update.
  loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
  params = update_params(params, learning_rate, grads)
  if i % n_print == 0:
    print(f'Loss step {i}: ', loss_val)
    plot_predictions(x_samples, y_samples, model, params)  

plot_predictions(x_samples, y_samples, model, params)  