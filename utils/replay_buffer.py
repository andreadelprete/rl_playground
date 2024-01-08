import jax.numpy as jnp

class ReplayBuffer:
    def __init__(self, name):
        self.name = name
        self.X = []
        self.Out = []
        self.Out_grad = []

    def append(self, x, out, out_grad=None):
        self.X.append(x)
        self.Out.append(out)
        if(out_grad is not None):
            self.Out_grad.append(out_grad)

    def getX(self):
        return jnp.array(self.X)
    
    def getOut(self):
        return jnp.array(self.Out).reshape((len(self.Out),1))
    
    def getOutGrad(self):
        return jnp.array(self.Out_grad) #.reshape((len(self.Out),1))
    
    def clean(self):
        self.X = []
        self.Out = []
        self.Out_grad = []