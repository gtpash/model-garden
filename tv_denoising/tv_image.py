import dolfin as dl
import ufl
import time
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

## boundaries for the unit square
def u0_boundary(x, on_boundary):
    return on_boundary

## constants
SEP = "\n"+"#"*80+"\n"
ALPHA = 1e-3  # weight for TV regularization
BETA = 1e-4  # mollifier for TV regularization
LUMPING = True
# PEPS = 1.  # full mass matrix
PEPS = 0.5*ALPHA  # mass matrix scaled with TV

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
nsprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA, peps=PEPS)

# set up the function spaces for the PDEProblem
Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
Vh = [Vhm, Vhm, Vhm]

ndofs = [Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()]
if rank == 0:
    print(SEP, "Set up the mesh and finite element spaces", SEP, flush=True)
    print(f"Number of dofs: STATE={ndofs[0]}, PARAMETER={ndofs[1]}, ADJOINT={ndofs[2]}", flush=True)

zero = dl.Constant(0.0)
bc = dl.DirichletBC(Vh[hp.STATE], zero, u0_boundary)  # homogeneous Dirichlet BC
bc0 = bc  # same for the adjoint

## define the variational form
if LUMPING:
    dx_lump = ufl.dx(scheme="vertex", metadata={"degree":1, "representation":"quadrature"})
    def pde_varf(u,m,p):
    # the parameter is the state (residual form)
        return u*p*dx_lump - m*p*dx_lump
else:
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

# set up the misfit (noise variance set to 1, since we don't scale misfit for image denoising)
misfit = hp.ContinuousStateObservation(Vh=Vh[hp.STATE], dX=ufl.dx, data=d.vector(), noise_variance=1., bcs=[bc])

# set up the prior
tvprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA)

# set up the model describing the inverse problem
TVonly = [True, False, True]
model = hp.ModelNS(pde, misfit, None, tvprior, which=TVonly)

# set up the solver and solve
m = dl.Function(Vhm)
m.vector().zero()
solver_params = hp.ReducedSpacePDNewtonCG_ParameterList()
solver_params["max_iter"] = 100
solver_params["cg_max_iter"] = 75
solver = hp.ReducedSpacePDNewtonCG(model, parameters=solver_params)

start = time.perf_counter()
x = solver.solve([None, m.vector(), None, None])
print(f"Nonlinear solve took:\t{(time.perf_counter()-start)/60:.2f} minutes", flush=True)

print("Solver convergence criterion", flush=True)
print(solver.termination_reasons[solver.reason])

xfunname = ["state", "parameter", "adjoint"]
xfun = [hp.vector2Function(x[i], Vh[i], name=xfunname[i]) for i in range(len(Vh))]

# show off your denoised image
plt.plot()
dl.plot(xfun[hp.PARAMETER])
plt.show()
