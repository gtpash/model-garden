import dolfin as dl
import ufl
import math
import numpy as np
import matplotlib.pyplot as plt

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
NX = 64
NY = 64
DIM = 2
NOISE_LEVEL = 0.02
ALPHA = 1e-2
BETA = 1e-4

## set up the mesh, mpi communicator, and function spaces
mesh = dl.UnitSquareMesh(NX, NY)
    
rank = dl.MPI.rank(mesh.mpi_comm())
nproc = dl.MPI.size(mesh.mpi_comm())

Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vh = [Vh2, Vh1, Vh2]

ndofs = [Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()]
if rank == 0:
    print(SEP, "Set up the mesh and finite element spaces", SEP)
    print(f"Number of dofs: STATE={ndofs[0]}, PARAMETER={ndofs[1]}, ADJOINT={ndofs[2]}")

## initialize forcing function, boundary conditions
f = dl.Constant(1.0)

zero = dl.Constant(0.0)
bc = dl.DirichletBC(Vh[hp.STATE], zero, u0_boundary)  # homogeneous Dirichlet BC
bc0 = bc  # same for the adjoint

## define the variational form

def pde_varf(u,m,p):
    return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx
    
pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

## set up the true parameter
mtrue_exp = dl.Expression('1.0 + 7.0*(x[0]<=0.8)*(x[0]>=0.2)*(x[1]<=0.8)*(x[1]>=0.2)', degree=1)
mtrue = dl.interpolate(mtrue_exp, Vh1)

## set up observation operator for the top right corner
xx = np.linspace(0.5, 1.0, 25, endpoint=False)
xv, yv = np.meshgrid(xx, xx)
targets = np.vstack([xv.ravel(), yv.ravel()]).T

if rank == 0:
    print(f"Number of observation points: {targets.shape[0]}")

## define observation operator
B = hp.assemblePointwiseObservation(Vh[hp.STATE], targets)
p2o = parameter2NoisyObservations(pde, mtrue.vector(), NOISE_LEVEL)
p2o.generateNoisyObservations()

## generate synthetic observations
# misfit = hp.DiscreteStateObservation(B, data, noise_std_dev**2)
misfit = hp.ContinuousStateObservation(Vh=Vh[hp.STATE], dX=ufl.dx, bcs=[bc])
misfit.d.axpy(1., p2o.noisy_data)
misfit.noise_variance = p2o.noise_std_dev**2

# smooth portion of the prior
# same as: https://github.com/hippylib/hippylib/blob/master/applications/poisson/model_subsurf.py

# anisotropic diffusion tensor
theta0 = 2.
theta1 = .5
alpha  = math.pi/4
anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree = 1)
anis_diff.set(theta0, theta1, alpha)

gamma = .1
delta = .5
prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, anis_diff, robin_bc=True)

# nonsmooth portion of the prior
Vhm = Vh1
Vhw = dl.VectorFunctionSpace(mesh, 'DG', 0)
Vhwnorm = dl.FunctionSpace(mesh, 'DG', 0)
nsprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA)

# set up the model describing the inverse problem
TVonly = [True, False, True]
model = hp.ModelNS(pde, misfit, prior, nsprior, which=TVonly)
# model = hp.ModelNS(pde, misfit, prior, nsprior)

m = prior.mean.copy()
m.zero()
solver = hp.ReducedSpacePDNewtonCG(model)
# x = solver.solve([None, m, None, nsprior.compute_w(m)])
x = solver.solve([None, m, None, None])
print("Solver convergence criterion")
print(solver.termination_reasons[solver.reason])

# todo: add visualization
xfunname = ["state", "parameter", "adjoint"]
xfun = [hp.vector2Function(x[i], Vh[i], name=xfunname[i]) for i in range(len(Vh))]

breakpoint()