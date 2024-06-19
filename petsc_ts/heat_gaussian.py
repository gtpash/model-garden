from mpi4py import MPI  # must be imported first or you get SCOTCH errors.

import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc  # needs to come in before dolfin.

import dolfin as dl  # last.
import ufl
import numpy as np

from TSVariationalProblem import TS_VariationalProblem

# blah blah header
# https://jsdokken.com/dolfinx-tutorial/chapter2/diffusion_code.html
# more demos: https://github.com/erdc/petsc4py/blob/master/demo/ode/

################################################################################
# 0. Set up any necessary expressions
################################################################################
u0_expr = dl.Expression("exp(-a * (x[0]*x[0] + x[1]*x[1]))", a=5, degree=2)

# define the residual form of the time-dependent PDE, F(u, udot, v) = 0
class Fvarf:
    def __init__(self, v, f):
        self.v = v  # test function
        self.f = f  # forcing function

    def __call__(self, t, u, u_t):
        form = ufl.inner(u_t, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
        
        return form

################################################################################
# 1. MPI setup
################################################################################
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
print(f"I'm rank {RANK} of {SIZE}", flush=True)

################################################################################
# 2. Variational problem setup
################################################################################
t0 = 0.         # simulation start time
tf = 1.         # simulation end time
num_steps = 50  # number of time steps

nx = 50         # number of mesh points in x-direction
ny = 50         # number of mesh points in y-direction

x1 = dl.Point(-2, -2)
x2 = dl.Point(2, 2)

mesh = dl.RectangleMesh(COMM, x1, x2, nx, ny)
V = dl.FunctionSpace(mesh, "CG", 1)

# Define the exact solution and boundary condition.
bc = dl.DirichletBC(V, dl.Constant(0.), "on_boundary")
u0 = dl.interpolate(u0_expr, V)

# Set up the variational problem.
f = dl.Constant(0.)  # no forcing.
u = dl.Function(V)
u_t = dl.Function(V)
v = dl.TestFunction(V)

Fvarf_handler = Fvarf(v, f)

################################################################################
# 3. Hook into PETSc.TS
################################################################################
problem = TS_VariationalProblem(Fvarf_handler, u_t, u0, [bc], G=None)
ts = PETSc.TS().create(comm=COMM)
ts.setType(PETSc.TS.Type.BEULER)
ts.setProblemType(PETSc.TS.ProblemType.LINEAR)
ts.setTimeSpan(np.linspace(t0, tf, num_steps + 1))  # +1 for book-keeping
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
# ts.setSaveTrajectory()

# you need to pass objects of the correct size
vec = dl.Function(V).vector().vec()  # PETSc Vec of appropriate size
M = dl.assemble( dl.inner(dl.TrialFunction(V), dl.TestFunction(V)) * dl.dx )
Jvec = dl.as_backend_type(M).mat()

ts.setIFunction(problem.evalFunction, vec)
ts.setIJacobian(problem.evalJacobian, Jvec)

################################################################################
# 4. Solve the problem
################################################################################
ts.setFromOptions()             # Apply run-time options, e.g. -ts_adapt_monitor -ts_type arkimex -snes_converged_reason
ts.solve(u0.vector().vec())

################################################################################
# 5. Report solutions
################################################################################
# import matplotlib.pyplot as plt
uh = dl.Function(V)

import os
os.makedirs("outputs", exist_ok=True)
with dl.XDMFFile(COMM, "outputs/heat_gaussian.xdmf") as fid:
    fid.parameters["functions_share_mesh"] = True
    fid.parameters["rewrite_function_mesh"] = False
    for i, sshot in enumerate(ts.getTimeSpanSolutions()):
        # load solution into a dolfin function
        uh.vector().zero()
        uh.vector().axpy(1., dl.PETScVector(sshot))
        
        t = ts.getTimeSpan()[i]      # time of snapshot
        fid.write(uh, t)             # write snapshot to file
