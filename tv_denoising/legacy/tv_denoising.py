'''
This script is an implementation of the CMIS lab code: https://github.com/uvilla/cmis_labs/blob/master/01_ImageDenoising/ImageDenoising_TV.ipynb
This script to verify the results of using total variation regularization.
This script is also used for development of hippylib codes.
'''

import logging
import numpy as np
import dolfin as dl
import scipy.io as sio
import matplotlib.pyplot as plt
import hippylib as hp
from unconstrainedMinimization import InexactNewtonCG

# Total Variation class
class TVDenoising:
    def __init__(self, V, d, alpha, beta):
        self.alpha   = dl.Constant(alpha)
        self.beta    = dl.Constant(beta)
        self.d       = d
        self.m_tilde  = dl.TestFunction(V)
        self.m_hat = dl.TrialFunction(V)
        
    def cost_reg(self, m):
        return dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)*dl.dx
    
    def cost_misfit(self, m):
        return dl.Constant(.5)*dl.inner(m-self.d, m - self.d)*dl.dx
        
    def cost(self, m):        
        return self.cost_misfit(m) + self.alpha*self.cost_reg(m)
        
    def grad(self, m):    
        grad_ls = dl.inner(self.m_tilde, m - self.d)*dl.dx
        
        TVm = dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)
        grad_tv = dl.Constant(1.)/TVm*dl.inner(dl.grad(m), dl.grad(self.m_tilde))*dl.dx
        
        grad = grad_ls + self.alpha*grad_tv
        
        return grad
        
    def Hessian(self,m):
        H_ls = dl.inner(self.m_tilde, self.m_hat)*dl.dx
        
        TVm = dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)
        A = dl.Constant(1.)/TVm * (dl.Identity(2) - dl.outer(dl.grad(m)/TVm, dl.grad(m)/TVm ) )
        H_tv = dl.inner(A*dl.grad(self.m_tilde), dl.grad(self.m_hat))*dl.dx
         
        H = H_ls + self.alpha*H_tv
                                   
        return H
    
    def LD_Hessian(self,m):
        # lagged diffusivity
        H_ls = dl.inner(self.m_tilde, self.m_hat)*dl.dx
        
        TVm = dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)
        H_tv = dl.Constant(1.)/TVm *dl.inner(dl.grad(self.m_tilde), dl.grad(self.m_hat))*dl.dx
         
        H = H_ls + self.alpha*H_tv
                                   
        return H

# Solve the problem with a given alpha and beta
def TVsolution(alpha, beta):
    
    m = dl.Function(Vh)
    problem = TVDenoising(Vh, d, alpha, beta)
    
    solver = InexactNewtonCG()
    solver.parameters["rel_tolerance"] = 1e-5
    solver.parameters["abs_tolerance"] = 1e-9
    solver.parameters["gdm_tolerance"] = 1e-18
    solver.parameters["max_iter"] = 1000
    solver.parameters["c_armijo"] = 1e-5
    solver.parameters["print_level"] = -1
    solver.parameters["max_backtracking_iter"] = 10
    solver.solve(problem.cost, problem.grad, problem.Hessian, m)
    
    MSE  = dl.inner(m - m_true, m - m_true)*dl.dx
    J    = problem.cost(m)
    J_ls = problem.cost_misfit(m)
    R_tv = problem.cost_reg(m)

    print( "{0:15e} {1:15e} {2:4d} {3:15e} {4:15e} {5:15e} {6:15e}".format(
           alpha, beta, solver.it, dl.assemble(J), dl.assemble(J_ls), dl.assemble(R_tv), dl.assemble(MSE))
         )

    return m


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

# Finite difference check of the TV gradient
n_eps = 32
eps = 1e-2*np.power(2., -np.arange(n_eps))
err_grad = np.zeros(n_eps)

m0 = dl.interpolate(dl.Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)", degree=4), Vh)
alpha = 1.
beta   = 1e-4
problem = TVDenoising(Vh, d, alpha, beta)

J0 = dl.assemble(problem.cost(m0))
grad0 = dl.assemble(problem.grad(m0))

mtilde = dl.Function(Vh)
mtilde.vector().set_local(np.random.randn(Vh.dim()))
mtilde.vector().apply("")

mtilde_grad0 = grad0.inner(mtilde.vector())

for i in range(n_eps):
    Jplus = dl.assemble( problem.cost(m0 + dl.Constant(eps[i])*mtilde) )
    err_grad[i] = abs( (Jplus - J0)/eps[i] - mtilde_grad0 )

plt.figure()    
plt.loglog(eps, err_grad, "-ob", label="Error Grad")
plt.loglog(eps, (.5*err_grad[0]/eps[0])*eps, "-.k", label="First Order")
plt.title("Finite difference check of the first variation (gradient)")
plt.xlabel("eps")
plt.ylabel("Error grad")
plt.legend(loc = "upper left")
plt.show()

# Finite difference check of the TV Hessian
H_0 = dl.assemble( problem.Hessian(m0) )
H_0mtilde = H_0 * mtilde.vector()
err_H = np.zeros(n_eps)

for i in range(n_eps):
    grad_plus = dl.assemble( problem.grad(m0 + dl.Constant(eps[i])*mtilde) )
    diff_grad = (grad_plus - grad0)
    diff_grad *= 1/eps[i]
    err_H[i] = (diff_grad - H_0mtilde).norm("l2")
    
plt.figure()    
plt.loglog(eps, err_H, "-ob", label="Error Hessian")
plt.loglog(eps, (.5*err_H[0]/eps[0])*eps, "-.k", label="First Order")
plt.title("Finite difference check of the second variation (Hessian)")
plt.xlabel("eps")
plt.ylabel("Error Hessian")
plt.legend(loc = "upper left")
plt.show()

# Compute the de-noised image for different regularization strengths.
print ("{0:15} {1:15} {2:4} {3:15} {4:15} {5:15} {6:15}".format("alpha", "beta", "nit", "J", "J_ls", "R_tv", "MSE") )

beta = 1e-4
n_alphas = 4
alphas = np.power(10., -np.arange(1, n_alphas+1))
for alpha in alphas:
    m = TVsolution(alpha, beta)
    plt.figure()
    dl.plot(m)
    plt.show()
