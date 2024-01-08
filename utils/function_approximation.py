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


@jax.jit
def update_params(params, learning_rate, grads):
  params = jax.tree_util.tree_map(
      lambda p, g: p - learning_rate * g, params, grads)
  return params


class NeuralNetwork:
    def __init__(self, name, in_dim, out_dim, layer_sizes, learning_rate=0.1, 
                 random_seed=0, sobolev_weight=0.1):
        # Set problem dimensions.
        self.name = name
        self.random_seed = random_seed
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate
        self.sobolev_weight = sobolev_weight

        self.model = MLP(layer_sizes + [out_dim])
        k1, k2 = random.split(random.key(random_seed))
        x = random.normal(k1, (in_dim,)) # Dummy input data
        self.params = self.model.init(k2, x) # Initialization call
        jax.tree_util.tree_map(lambda x: x.shape, self.params) # Checking output shapes

        # Same as JAX version but using model.apply().
        @jax.jit
        def mse(params, x_batched, y_batched):
            # Define the squared loss for a single pair (x,y)
            def squared_error(x, y):
                pred = self.model.apply(params, x)
                return jnp.inner(y-pred, y-pred) / 2.0
            # Vectorize the previous to compute the average of the loss on all samples.
            return jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0)
        
        self.loss_grad_fn = jax.value_and_grad(mse)
        self.loss = mse

        @jax.jit
        def mse_sobolev(params, x_batched, y_batched, dydx_batched):
            def network_scalar(x):
                return self.model.apply(params, x)[0]
            
            # Define the squared loss for a single tuple (x,y, dydx)
            def squared_error_sobolev(x, y, dydx):
                # We should use jacfwd if network's output was not a scalar
                # critic_grad = jax.jacfwd(critic)
                # dVdx = jax.vmap(critic_grad)(X_aug)[:,0,0]
                pred_grad = jax.grad(network_scalar)
                e_y  = y    - self.model.apply(params, x)
                e_dy = dydx - pred_grad(x)[0]
                return jnp.inner(e_y, e_y)/2.0 + self.sobolev_weight*jnp.inner(e_dy, e_dy)/2.0
                # return jnp.inner(e_y, e_y)/2.0 + self.sobolev_weight*jnp.log(jnp.inner(e_dy, e_dy))
            # Vectorize the previous to compute the average of the loss on all samples.
            return jnp.mean(jax.vmap(squared_error_sobolev)(x_batched,y_batched,dydx_batched), axis=0)
        self.loss_sob_grad_fn = jax.value_and_grad(mse_sobolev)
        self.loss_sob = mse_sobolev

    def __call__(self, x):
       return self.model.apply(self.params, x)

    def train(self, in_samples, out_samples, n_iter=1, minibatch_size=32, learning_rate=None):
        if(learning_rate is None):
           learning_rate = self.learning_rate
        
        N = in_samples.shape[0]
        assert(out_samples.shape[0]==N)
        for i in range(n_iter): # Perform one gradient update.    
            ind = np.random.randint(N, size=minibatch_size)
            loss_val, grads = self.loss_grad_fn(self.params, in_samples[ind,:], out_samples[ind,:])
            self.params = update_params(self.params, learning_rate, grads)
        return self.loss(self.params, in_samples, out_samples)
    
    def train_sobolev(self, in_samples, out_samples, out_grad_samples, n_iter=1, minibatch_size=32, learning_rate=None):
        if(learning_rate is None):
           learning_rate = self.learning_rate
        
        N = in_samples.shape[0]
        assert(out_samples.shape[0]==N)
        assert(out_grad_samples.shape[0]==N)
        for i in range(n_iter): # Perform one gradient update.    
            ind = np.random.randint(N, size=minibatch_size)
            loss_val, grads = self.loss_sob_grad_fn(self.params, in_samples[ind,:], 
                                                    out_samples[ind,:], out_grad_samples[ind,:])
            self.params = update_params(self.params, learning_rate, grads)
        return self.loss_sob(self.params, in_samples, out_samples, out_grad_samples)


class ActorMLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.elu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    # x = nn.tanh(nn.Dense(self.features[-1])(x))
    return x
  
class ActorNetwork:
    def __init__(self, name, in_dim, out_dim, layer_sizes, cost, V_model, dynamic, 
                 learning_rate=0.1, random_seed=0):
        # Set problem dimensions.
        self.name = name
        self.random_seed = random_seed
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate

        self.model = ActorMLP(layer_sizes + [out_dim])
        self.V_model = V_model
        self.dynamic = dynamic
        self.cost = cost

        k1, k2 = random.split(random.key(random_seed))
        x = random.normal(k1, (in_dim,)) # Dummy input data
        self.params = self.model.init(k2, x) # Initialization call
        jax.tree_util.tree_map(lambda x: x.shape, self.params) # Checking output shapes
        
        # Same as JAX version but using model.apply().
        # @jax.jit
        # def Q_batch(params, x_batched):           
        #     # Define the loss for a single x
        #     def Q_single(x):
        #         u = self.model.apply(params, x)
        #         return self.cost(x, u) + self.V_model(self.dynamic(x,u))[0]
        #     # Vectorize the previous to compute the average of the loss on all samples.
        #     return jnp.mean(jax.vmap(Q_single)(x_batched), axis=0)
        
        # self.loss_grad_fn = jax.value_and_grad(Q_batch)
        # self.loss = Q_batch

        @jax.jit
        def mse(params, x_batched, y_batched):
            # Define the squared loss for a single pair (x,y)
            def squared_error(x, y):
                pred = self.model.apply(params, x)
                return jnp.inner(y-pred, y-pred) / 2.0
            # Vectorize the previous to compute the average of the loss on all samples.
            return jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0)
        
        self.loss_grad_mse = jax.value_and_grad(mse)
        self.loss_mse = mse

    def __call__(self, x):
       return self.model.apply(self.params, x)
    
    def train(self, in_samples, n_iter=1, minibatch_size=32, learning_rate=None):
        if(learning_rate is None):
           learning_rate = self.learning_rate

        def Q_batch(params, x_batched):           
            def Q_single(x):
                u = self.model.apply(params, x)
                return self.cost(x, u) + self.V_model(self.dynamic(x,u))[0]
            return jnp.mean(jax.vmap(Q_single)(x_batched), axis=0)
        self.loss_grad_fn = jax.jit(jax.value_and_grad(Q_batch))
        self.loss = Q_batch

        N = in_samples.shape[0]
        for i in range(n_iter):
            # Perform one gradient update.            
            ind = np.random.randint(N, size=minibatch_size)
            loss_val, grads = self.loss_grad_fn(self.params, in_samples[ind,:])
            self.params = update_params(self.params, learning_rate, grads)
        return self.loss(self.params, in_samples)

    def train_supervised(self, in_samples, out_samples, n_iter=1, minibatch_size=32, learning_rate=None):
        if(learning_rate is None):
           learning_rate = self.learning_rate
        
        N = in_samples.shape[0]
        for i in range(n_iter):
            # Perform one gradient update.    
            ind = np.random.randint(N, size=minibatch_size)
            loss_val, grads = self.loss_grad_mse(self.params, in_samples[ind,:], out_samples[ind,:])
            self.params = update_params(self.params, learning_rate, grads)
        return self.loss_mse(self.params, in_samples, out_samples)