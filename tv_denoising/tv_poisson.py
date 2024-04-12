import dolfin as dl
import ufl
import matplotlib.pyplot as plt
import time

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
PEPS = 0.5*ALPHA  # mass matrix scaled with TV
MAX_ITER = 100
CG_MAX_ITER = 75

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

plt.figure()
dl.plot(mtrue)
plt.show()

## define observation operator
p2o = parameter2NoisyObservations(pde, mtrue.vector(), NOISE_LEVEL)
p2o.generateNoisyObservations()

## generate synthetic observations
misfit = hp.ContinuousStateObservation(Vh=Vh[hp.STATE], dX=ufl.dx, data=p2o.noisy_data, noise_variance=p2o.noise_std_dev**2, bcs=[bc])

# nonsmooth portion of the prior
Vhm = Vh1
Vhw = dl.VectorFunctionSpace(mesh, 'DG', 0)
Vhwnorm = dl.FunctionSpace(mesh, 'DG', 0)
nsprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA)

# set up the model describing the inverse problem
TVonly = [True, False, True]
model = hp.ModelNS(pde, misfit, None, nsprior, which=TVonly)

solver_params = hp.ReducedSpacePDNewtonCG_ParameterList()
solver_params["max_iter"] = MAX_ITER
solver_params["cg_max_iter"] = CG_MAX_ITER

solver = hp.ReducedSpacePDNewtonCG(model, parameters=solver_params)

# initial guess (zero)
m = pde.generate_parameter()
m.zero()

# solve the system
start = time.perf_counter()
x = solver.solve([None, m, None, None])
print(f"Time to solve: {(time.perf_counter()-start)/60:.2f} minutes")
print(f"Solver convergence criterion:\t{solver.termination_reasons[solver.reason]}")

# extract the solution and plot it
xfunname = ["state", "parameter", "adjoint"]
xfun = [hp.vector2Function(x[i], Vh[i], name=xfunname[i]) for i in range(len(Vh))]

plt.figure()
dl.plot(xfun[hp.PARAMETER])
plt.show()
