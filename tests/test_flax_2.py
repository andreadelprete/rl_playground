from rl_playground.utils.function_approximation import NeuralNetwork
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Set problem dimensions.
RANDOM_KEY = 3
n_samples = 100
x_dim = 1
y_dim = 1
learning_rate = 0.1  # Gradient step size.
n_iter = 1000
function_type = 'sin' # linear or sin_


# Generate samples with additional noise.
k1, k2 = random.split(random.key(RANDOM_KEY))
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


def plot_predictions(x_samples, y_samples, model, params):
  plt.plot(x_samples, y_samples, 'xr ', label='Training data')
  y_pred = model.apply(params, x_samples)
  plt.plot(x_samples, y_pred, 'xk ', label='Prediction')
  plt.legend()
  plt.show()
  
critic_layers = [32, 32]
nn = NeuralNetwork("Value", x_dim, y_dim, critic_layers, learning_rate, RANDOM_KEY)
for i in range(10):
    nn.train(x_samples, y_samples, int(n_iter/10))
    # print(x_samples.shape)
    # print(y_samples.shape)
    plot_predictions(x_samples, y_samples, nn.model, nn.params)
