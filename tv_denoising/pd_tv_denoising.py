'''
This script is an implementation of the CMIS lab code: https://github.com/uvilla/cmis_labs/blob/master/01_ImageDenoising/ImageDenoising_PrimalDualTV.ipynb
This script to verify the results of using total variation regularization.
This script is also used for development of hippylib codes.
'''

import math
import logging
import numpy as np
import dolfin as dl
import scipy.io as sio
import matplotlib.pyplot as plt
import hippylib as hp

class PDTVDenoising:
    def __init__(self, Vm, Vw, Vwnorm, d, alpha, beta):
        self.alpha   = dl.Constant(alpha)
        self.beta    = dl.Constant(beta)
        self.d       = d
        self.m_tilde  = dl.TestFunction(Vm)
        self.m_hat = dl.TrialFunction(Vm)
        
        self.Vm = Vm
        self.Vw = Vw
        self.Vwnorm = Vwnorm
        
    def cost_reg(self, m):
        return dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)*dl.dx
    
    def cost_misfit(self, m):
        return dl.Constant(.5)*dl.inner(m-self.d, m - self.d)*dl.dx
        
    def cost(self, m):        
        return self.cost_misfit(m) + self.alpha*self.cost_reg(m)
        
    def grad_m(self, m):    
        grad_ls = dl.inner(self.m_tilde, m - self.d)*dl.dx        
        TVm = dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)
        grad_tv = dl.Constant(1.)/TVm*dl.inner(dl.grad(m), dl.grad(self.m_tilde))*dl.dx
        
        grad = grad_ls + self.alpha*grad_tv
        
        return grad
        
    def Hessian(self,m, w):
        H_ls = dl.inner(self.m_tilde, self.m_hat)*dl.dx
        
        TVm = dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)
        A = dl.Constant(1.)/TVm * (dl.Identity(2) 
                                   - dl.Constant(.5)*dl.outer(w, dl.grad(m)/TVm )
                                   - dl.Constant(.5)*dl.outer(dl.grad(m)/TVm, w ) )
        
        H_tv = dl.inner(A*dl.grad(self.m_tilde), dl.grad(self.m_hat))*dl.dx
         
        H = H_ls + self.alpha*H_tv
                                   
        return H
    
    def compute_w_hat(self, m, w, m_hat):
        TVm = dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)
        A = dl.Constant(1.)/TVm * (dl.Identity(2) 
                                   - dl.Constant(.5)*dl.outer(w, dl.grad(m)/TVm )
                                   - dl.Constant(.5)*dl.outer(dl.grad(m)/TVm, w ) )
        
        expression = A*dl.grad(m_hat) - w + dl.grad(m)/TVm
        
        return dl.project(expression, self.Vw)
    
    def wnorm(self, w):
        return dl.inner(w,w)

# Suppress FEniCS output
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

# Load in the data.
data = sio.loadmat("circles.mat")["im"]
Lx = 1.
h = Lx/float(data.shape[0])
Ly = float(data.shape[1])*h

mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(Lx, Ly), data.shape[0], data.shape[1])
Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
true = hp.NumpyScalarExpression2D()
true.setData(data, h, h)
m_true = dl.interpolate(true, Vh)

# Add noise to the image.
np.random.seed(1)
noise_stddev = 0.3
noise = noise_stddev*np.random.randn(*data.shape)
noisy = hp.NumpyScalarExpression2D()
noisy.setData(data + noise, h, h)
d = dl.interpolate(noisy, Vh)

# For sclaing.
vmin = np.min(d.vector().get_local())
vmax = np.max(d.vector().get_local())

plt.figure()
dl.plot(d)
plt.show()
dl.plot(m_true)
plt.show()


