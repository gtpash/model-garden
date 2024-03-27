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

from utils import generateNoisyPointwiseObservations

## boundaries for the unit square
def x_boundary(x, on_boundary):
    return on_boundary and (x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

def y_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def u0_boundary(x, on_boundary):
    return on_boundary

## constants
NX = 64
NY = 64
DIM = 2
NOISE_LEVEL = 0.02
SEP = "\n"+"#"*80+"\n"

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

## generate synthetic observations
data, noise_std_dev = generateNoisyPointwiseObservations(pde, B, mtrue.vector(), NOISE_LEVEL)
breakpoint()

misfit = hp.DiscreteStateObservation(B, data, noise_std_dev**2)

model = hp.Model(pde, prior, misfit)

breakpoint()