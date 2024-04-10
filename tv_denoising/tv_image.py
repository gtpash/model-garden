import dolfin as dl
import ufl
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_DEV_PATH') )
import hippylib as hp

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

from utils import parameter2NoisyObservations

## boundaries for the unit square
def x_boundary(x, on_boundary):
    return on_boundary and (x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

def y_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def u0_boundary(x, on_boundary):
    return on_boundary

## constants
SEP = "\n"+"#"*80+"\n"
ALPHA = 1e-3
BETA = 1e-4

## set up the mesh, mpi communicator, and function spaces
img = sio.loadmat("circles.mat")["im"]
Lx = 1.
h = Lx/float(img.shape[0])
Ly = float(img.shape[1])*h

mesh = dl.RectangleMesh(dl.Point(0., 0.), dl.Point(Lx, Ly), img.shape[0], img.shape[1])

rank = dl.MPI.rank(mesh.mpi_comm())
nproc = dl.MPI.size(mesh.mpi_comm())

# set up the TV prior
Vhm = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vhw = dl.VectorFunctionSpace(mesh, 'DG', 0)
Vhwnorm = dl.FunctionSpace(mesh, 'DG', 0)
nsprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA)

# set up the function spaces for the PDEProblem
Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
Vh = [Vhm, Vhm, Vhm]

ndofs = [Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()]
if rank == 0:
    print(SEP, "Set up the mesh and finite element spaces", SEP)
    print(f"Number of dofs: STATE={ndofs[0]}, PARAMETER={ndofs[1]}, ADJOINT={ndofs[2]}")

zero = dl.Constant(0.0)
bc = dl.DirichletBC(Vh[hp.STATE], zero, u0_boundary)  # homogeneous Dirichlet BC
bc0 = bc  # same for the adjoint

## define the variational form
def pde_varf(u,m,p):
    # the parameter is the state (residual form)
    return u*p*ufl.dx - m*p*ufl.dx
    
pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

## set up the true parameter
true = hp.NumpyScalarExpression2D()
true.setData(img, h, h)
m_true = dl.interpolate(true, Vhm)

# add noise to the image
np.random.seed(1)
noise_stddev = 0.3
noise = noise_stddev*np.random.randn(*img.shape)
noisy = hp.NumpyScalarExpression2D()
noisy.setData(img + noise, h, h)
d = dl.interpolate(noisy, Vhm)

# for scaling
vmin = np.min(d.vector().get_local())
vmax = np.max(d.vector().get_local())

# show the images
plt.plot()
dl.plot(d)
plt.show()

plt.plot()
dl.plot(m_true)
plt.show()

# set up the misfit
misfit = hp.ContinuousStateObservation(Vh=Vh[hp.STATE], dX=ufl.dx, bcs=[bc])
misfit.d.axpy(1., d.vector())
misfit.noise_variance = noise_stddev**2

# set up the prior
tvprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA)

# set up the model describing the inverse problem
TVonly = [True, False, True]
model = hp.ModelNS(pde, misfit, None, tvprior, which=TVonly)

# set up the solver and solve
m = dl.Function(Vhm)
m.vector().zero()
solver = hp.ReducedSpacePDNewtonCG(model)
x = solver.solve([None, m.vector(), None, None])

print("Solver convergence criterion")
print(solver.termination_reasons[solver.reason])

# todo: add visualization
xfunname = ["state", "parameter", "adjoint"]
xfun = [hp.vector2Function(x[i], Vh[i], name=xfunname[i]) for i in range(len(Vh))]

breakpoint()