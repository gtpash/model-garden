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
# https://jsdokken.com/dolfinx-tutorial/chapter2/heat_code.html
# more demos: https://github.com/erdc/petsc4py/blob/master/demo/ode/

################################################################################
# 0. Set up the exact solution, boundary conditions
################################################################################
class exactSolExpression(dl.UserExpression):
    def __init__(self, alpha, beta, t, **kwargs):
        super().__init__(kwargs)
        
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def eval(self, value, x):
        value[0] = 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t

    def value_shape(self):
        return ()
    
def uex(alpha, beta, t):
    return dl.Expression("1 + x[0]*x[0] + alpha * x[1]*x[1] + beta * t", alpha=alpha, beta=beta, t=t, degree=2)
    
class Boundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and (dl.near(x[0], 0, tol) or dl.near(x[0], 1, tol) or dl.near(x[1], 0, tol) or dl.near(x[1], 1, tol))
    
def boundary(x, on_boundary):
    tol = 1E-14
    return on_boundary and (dl.near(x[0], 0, tol) or dl.near(x[0], 1, tol) or dl.near(x[1], 0, tol) or dl.near(x[1], 1, tol))

import matplotlib.pyplot as plt
    
class exactSolBC():
    def __init__(self, alpha, beta, V, boundary):
        self.alpha = alpha
        self.beta = beta
        self.V = V
        self.boundary = boundary
        
    def __call__(self, t, u, udot):
        # this is where you'd define the time-dependent BC.
        bc_expr = dl.Expression("1 + x[0]*x[0] + alpha * x[1]*x[1] + beta * t", alpha=alpha, beta=beta, t=t, degree=2)
        u_D = dl.interpolate(bc_expr, self.V)
        # dl.plot(u_D)
        # plt.savefig(f"outputs/heat_bc_{t}.png")
        # plt.close()
        return dl.DirichletBC(self.V, u_D, self.boundary)
        # return dl.DirichletBC(self.V, u_D, dl.DomainBoundary())
    

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
tf = 2.         # simulation end time
num_steps = 20  # number of time steps
alpha = 3       # simulation parameter
beta = 1.2      # simulation parameter

nx = 5          # number of mesh points in x-direction
ny = 5          # number of mesh points in y-direction

mesh = dl.UnitSquareMesh(COMM, 10, 10)
V = dl.FunctionSpace(mesh, "CG", 1)

# Define the exact solution and boundary condition.
u_exact_epxr = lambda t: exactSolExpression(alpha, beta, t)
bc_handler = exactSolBC(alpha, beta, V, boundary)
# u0 = dl.interpolate(u_exact_epxr(t0), V)
u0 = dl.interpolate(uex(alpha, beta, t0), V)

# Set up the variational problem.
f = dl.Constant( beta - 2 - 2 * alpha )  # forcing function
u = dl.Function(V)
u_t = dl.Function(V)
v = dl.TestFunction(V)

Fvarf_handler = Fvarf(v, f)

################################################################################
# 3. Hook into PETSc.TS
################################################################################
problem = TS_VariationalProblem(Fvarf_handler, u_t, u0, [bc_handler], G=None)
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
# 5. Compute L2 error at last step
################################################################################
V_ex = dl.FunctionSpace(mesh, "Lagrange", 2)  # higher order space for exact solution
u_ex = dl.interpolate(uex(alpha, beta, tf), V_ex)


################################################################################
# 5. Report solutions
################################################################################
# import matplotlib.pyplot as plt
uh = dl.Function(V)

import os
os.makedirs("outputs", exist_ok=True)
with dl.XDMFFile(COMM, "outputs/heat.xdmf") as fid:
    fid.parameters["functions_share_mesh"] = True
    fid.parameters["rewrite_function_mesh"] = False
    for i, sshot in enumerate(ts.getTimeSpanSolutions()):
        # load solution into a dolfin function
        uh.vector().zero()
        uh.vector().axpy(1., dl.PETScVector(sshot))
        
        t = ts.getTimeSpan()[i]      # time of snapshot
        fid.write(uh, t)             # write snapshot to file
        
with dl.XDMFFile(COMM, "outputs/heat_ex.xdmf") as fid:
    fid.parameters["functions_share_mesh"] = True
    fid.parameters["rewrite_function_mesh"] = False
    for i, t in enumerate(ts.getTimeSpan()):
        # load solution into a dolfin function
        u_ex = dl.interpolate(uex(alpha, beta, t), V_ex)
        fid.write(uh, t)             # write snapshot to file
        
for i, t in enumerate(ts.getTimeSpan()):
    # load solution into a dolfin function
    u_ex = dl.interpolate(uex(alpha, beta, t), V_ex)
    dl.plot(u_ex)
    plt.savefig(f"outputs/heat_ex_{t}.png")
    plt.close()


breakpoint()
import matplotlib.pyplot as plt

uh = dl.Function(V)
uh.vector().axpy(1., dl.PETScVector(ts.getSolution()))

residual = uh - u_ex
residual = dl.assemble( dl.inner(residual, residual) * dl.dx )

error_L2 = np.sqrt(COMM.allreduce(residual, op=MPI.SUM))  # todo: is this mass weighted L^2(\Omega) or just \ell^2 ?
if RANK == 0:
    print(f"L2-error: {error_L2:.2e}", flush=True)

breakpoint()

# # ## the mass matrix
# mform = dolfin.inner(v, u)*dolfin.dx
# massm = dolfin.assemble(mform)
# mmat = dolfin.as_backend_type(massm).sparray()
# mmat.eliminate_zeros()
# # factorize it for later
# mfac = SparseFactorMassmat(mmat)

# # norm induced by the mass matrix == discrete L2-norm
# def mnorm(uvec):
#     return np.sqrt(np.inner(uvec, mmat.dot(uvec)))


# error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))

# # Compute values at mesh vertices
# error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
# if RANK == 0:
#     print(f"Error_max: {error_max:.2e}", flush=True)
