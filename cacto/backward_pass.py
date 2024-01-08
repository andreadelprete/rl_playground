#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward pass of the Differential Dynamic Programming algorithm. 
Two versions are available, one considering box constraints on the control inputs.
JAX autodiff functions are used to differentiate cost and dynamics.

@author: adelprete
"""
import numpy as np
import jax
import jax.numpy as jnp

    
def backward_pass(X_bar, U_bar, cost, dynamic, mu):
    n = X_bar.shape[1]
    m = U_bar.shape[1]
    N = U_bar.shape[0]
    rx = list(range(0,n))
    ru = list(range(0,m))
    
    # the Value function is defined by a quadratic function: 0.5 x' V_{xx,i} x + V_{x,i} x
    V_xx = np.zeros((N+1, n, n))
    V_x  = np.zeros((N+1, n))
    
    # dynamics derivatives w.r.t. x and u
    A = np.zeros((N, n, n))
    B = np.zeros((N, n, m))
    
    # the task is defined by a quadratic cost: 
    # 0.5 x' l_{xx,i} x + l_{x,i} x +  0.5 u' l_{uu,i} u + l_{u,i} u + x' l_{xu,i} u
    l_x = np.empty((N+1, n))
    l_u = np.empty((N+1, m))
    l_xx = np.empty((N+1, n, n))
    l_uu = np.empty((N+1, m, m))
    Q_x = np.empty((N+1, n))
    Q_u = np.empty((N+1, m))
    Q_xx = np.empty((N+1, n, n))
    Q_uu = np.empty((N+1, m, m))
    Q_xu = np.empty((N+1, n, m))
    w = np.empty((N+1, m))
    K = np.empty((N+1, m, n))

    cost_final_x = jax.grad(cost, argnums=0)
    cost_final_xx = jax.hessian(cost, argnums=0)
    cost_running_x = jax.grad(cost, argnums=0)
    cost_running_u = jax.grad(cost, argnums=1)
    cost_running_xx = jax.hessian(cost, argnums=0)
    cost_running_uu = jax.hessian(cost, argnums=1)
    f_x = jax.jacrev(dynamic, argnums=0)
    f_u = jax.jacrev(dynamic, argnums=1)
    
    # initialize value function
    l_x[-1,:]  = cost_final_x(X_bar[-1,:], jnp.zeros(m))
    l_xx[-1,:,:] = cost_final_xx(X_bar[-1,:], jnp.zeros(m))
    V_xx[N,:,:] = l_xx[N,:,:]
    V_x[N,:]    = l_x[N,:]
    
    for i in range(N-1, -1, -1):                
        # compute dynamics Jacobians
        A[i,:,:] = f_x(X_bar[i,:], U_bar[i,:])
        B[i,:,:] = f_u(X_bar[i,:], U_bar[i,:])
            
        # compute the gradient of the cost function at X=X_bar
        l_x[i,:]    = cost_running_x(X_bar[i,:], U_bar[i,:])
        l_xx[i,:,:] = cost_running_xx(X_bar[i,:], U_bar[i,:])
        l_u[i,:]    = cost_running_u(X_bar[i,:], U_bar[i,:])
        l_uu[i,:,:] = cost_running_uu(X_bar[i,:], U_bar[i,:])
        # l_xu[i,:,:] = cost_running_xu(X_bar[i,:], U_bar[i,:])
        
        # compute regularized cost-to-go
        Q_x[i,:]     = l_x[i,:] + A[i,:,:].T @ V_x[i+1,:]
        Q_u[i,:]     = l_u[i,:] + B[i,:,:].T @ V_x[i+1,:]
        Q_xx[i,:,:]  = l_xx[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ A[i,:,:]
        Q_uu[i,:,:]  = l_uu[i,:,:] + B[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
        Q_xu[i,:,:]  = A[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
                        
        Qbar_uu       = Q_uu[i,:,:] + mu*np.identity(m)
        Qbar_uu_pinv  = np.linalg.pinv(Qbar_uu)
        w[i,:]       = - Qbar_uu_pinv @ Q_u[i,:]
        K[i,:,:]     = - Qbar_uu_pinv @ Q_xu[i,:,:].T
            
        # update Value function
        V_x[i,:]    = (Q_x[i,:] + K[i,:,:].T @ Q_u[i,:] +
            K[i,:,:].T @ Q_uu[i,:,:] @ w[i,:] + Q_xu[i,:,:] @ w[i,:])
        V_xx[i,:]   = (Q_xx[i,:,:] + K[i,:,:].T @ Q_uu[i,:,:] @ K[i,:,:] + 
            Q_xu[i,:,:] @ K[i,:,:] + K[i,:,:].T @ Q_xu[i,:,:].T)
                
    return V_x

def backward_pass_box(X_bar, U_bar, cost, dynamic, mu, u_min, u_max):
    ''' Backward pass of box DDP, which is DDP with control bounds. '''
    n = X_bar.shape[1]
    m = U_bar.shape[1]
    N = U_bar.shape[0]
    rx = list(range(0,n))
    ru = list(range(0,m))
    
    # the Value function is defined by a quadratic function: 0.5 x' V_{xx,i} x + V_{x,i} x
    V_xx = np.zeros((N+1, n, n))
    V_x  = np.zeros((N+1, n))
    
    # dynamics derivatives w.r.t. x and u
    A = np.zeros((N, n, n))
    B = np.zeros((N, n, m))
    
    # the task is defined by a quadratic cost: 
    # 0.5 x' l_{xx,i} x + l_{x,i} x +  0.5 u' l_{uu,i} u + l_{u,i} u + x' l_{xu,i} u
    l_x = np.empty((N+1, n))
    l_u = np.empty((N+1, m))
    l_xx = np.empty((N+1, n, n))
    l_uu = np.empty((N+1, m, m))
    Q_x = np.empty((N+1, n))
    Q_u = np.empty((N+1, m))
    Q_xx = np.empty((N+1, n, n))
    Q_uu = np.empty((N+1, m, m))
    Q_xu = np.empty((N+1, n, m))
    w = np.empty((N+1, m))
    K = np.empty((N+1, m, n))

    cost_final_x = jax.grad(cost, argnums=0)
    cost_final_xx = jax.hessian(cost, argnums=0)
    cost_running_x = jax.grad(cost, argnums=0)
    cost_running_u = jax.grad(cost, argnums=1)
    cost_running_xx = jax.hessian(cost, argnums=0)
    cost_running_uu = jax.hessian(cost, argnums=1)
    f_x = jax.jacrev(dynamic, argnums=0)
    f_u = jax.jacrev(dynamic, argnums=1)
    
    # initialize value function
    l_x[-1,:]  = cost_final_x(X_bar[-1,:], jnp.zeros(m))
    l_xx[-1,:,:] = cost_final_xx(X_bar[-1,:], jnp.zeros(m))
    V_xx[N,:,:] = l_xx[N,:,:]
    V_x[N,:]    = l_x[N,:]
    
    for i in range(N-1, -1, -1):                
        # compute dynamics Jacobians
        A[i,:,:] = f_x(X_bar[i,:], U_bar[i,:])
        B[i,:,:] = f_u(X_bar[i,:], U_bar[i,:])
            
        # compute the gradient of the cost function at X=X_bar
        l_x[i,:]    = cost_running_x(X_bar[i,:], U_bar[i,:])
        l_xx[i,:,:] = cost_running_xx(X_bar[i,:], U_bar[i,:])
        l_u[i,:]    = cost_running_u(X_bar[i,:], U_bar[i,:])
        l_uu[i,:,:] = cost_running_uu(X_bar[i,:], U_bar[i,:])
        # l_xu[i,:,:] = cost_running_xu(X_bar[i,:], U_bar[i,:])
        
        # compute regularized cost-to-go
        Q_x[i,:]     = l_x[i,:] + A[i,:,:].T @ V_x[i+1,:]
        Q_u[i,:]     = l_u[i,:] + B[i,:,:].T @ V_x[i+1,:]
        Q_xx[i,:,:]  = l_xx[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ A[i,:,:]
        Q_uu[i,:,:]  = l_uu[i,:,:] + B[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
        Q_xu[i,:,:]  = A[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
                        
        # account for directions in which u is saturated
        i_bool = np.logical_and(U_bar[i,:]<u_max, U_bar[i,:]>u_min)
        m_free = int(np.sum(i_bool)) # number of free control inputs
        # print("Max |u| = ", np.max(np.abs(U_bar[i])))
        # print("Number of saturated control inputs: ", m-m_free)
        i_free = np.where(i_bool)[0] # indeces of the free control inputs
        Q_uu_i = Q_uu[i,:,:]
        # create a reduced Q_uu considering only the non-saturated (free) control inputs
        Q_uu_free       = Q_uu_i[i_bool,:][:,i_bool] + mu*np.identity(m_free)
        Q_uu_free_inv   = np.linalg.pinv(Q_uu_free)
        # map Q_uu_free_inv to Q_uu_inv
        Q_uu_inv = np.zeros((m,m))
        for j in range(m_free):
            for k in range(m_free):
                Q_uu_inv[i_free[j], i_free[k]] = Q_uu_free_inv[j,k]
        w[i,:]       = - Q_uu_inv @ Q_u[i,:]
        K[i,:,:]     = - Q_uu_inv @ Q_xu[i,:,:].T
            
        # update Value function
        V_x[i,:]    = (Q_x[i,:] + K[i,:,:].T @ Q_u[i,:] +
            K[i,:,:].T @ Q_uu[i,:,:] @ w[i,:] + Q_xu[i,:,:] @ w[i,:])
        V_xx[i,:]   = (Q_xx[i,:,:] + K[i,:,:].T @ Q_uu[i,:,:] @ K[i,:,:] + 
            Q_xu[i,:,:] @ K[i,:,:] + K[i,:,:].T @ Q_xu[i,:,:].T)
                
    return V_x
