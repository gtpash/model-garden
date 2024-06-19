from mpi4py import MPI  # must be imported first or you get SCOTCH errors.

import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc  # needs to come in before dolfin.


import dolfin as dl  # last.
import ufl


# blah blah header
# https://jsdokken.com/dolfinx-tutorial/chapter2/heat_code.html
# more demos: https://github.com/erdc/petsc4py/blob/master/demo/ode/


'''
def check_pde_args(F, G, J, Jp):
    if not isinstance(F, ufl.form.Form):
        raise TypeError(f"Provided residual is a '{type(F).__name__}', not a Form")
    if len(F.arguments()) != 1:
        raise ValueError("Provided residual is not a linear form")
    if G is not None and not isinstance(G, (ufl.BaseForm, slate.TensorBase)):
        raise TypeError(f"Provided G residual is a '{type(G).__name__}', not a BaseForm or Slate Tensor")
    if G is not None and len(G.arguments()) != 1:
        raise ValueError("Provided G residual is not a linear form")
    if not isinstance(J, (ufl.BaseForm, slate.TensorBase)):
        raise TypeError("Provided Jacobian is a '%s', not a BaseForm or Slate Tensor" % type(J).__name__)
    if len(J.arguments()) != 2:
        raise ValueError("Provided Jacobian is not a bilinear form")
    if Jp is not None and not isinstance(Jp, (ufl.BaseForm, slate.TensorBase)):
        raise TypeError("Provided preconditioner is a '%s', not a BaseForm or Slate Tensor" % type(Jp).__name__)
    if Jp is not None and len(Jp.arguments()) != 2:
        raise ValueError("Provided preconditioner is not a bilinear form")

class DAEProblem(object):
    r"""Nonlinear variational problem in DAE form F(u̇, u, t) = G(u, t)."""

    def __init__(
        self,
        F,
        u,
        udot,
        tspan,
        time=None,
        bcs=None,
        J=None,
        Jp=None,
        form_compiler_parameters=None,
        is_linear=False,
        G=None,
    ):
        r"""
        :param F: the nonlinear form
        :param u: the :class:`.Function` to solve for
        :param udot: the :class:`.Function` for time derivative
        :param tspan: the tuple for start time and end time
        :param time: the :class:`.Constant` for time-dependent weak forms
        :param bcs: the boundary conditions (optional)
        :param J: the Jacobian J = sigma*dF/du̇ + dF/du (optional)
        :param Jp: a form used for preconditioning the linear system,
                 optional, if not supplied then the Jacobian itself
                 will be used.
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        :is_linear: internally used to check if all domain/bc forms
            are given either in 'A == b' style or in 'F == 0' style.
        :param G: G(t, u) term that will be treated explicitly
            when using an IMEX method for solving F(u̇, u, t) = G(u, t).
            If G is `None` the G(u, t) term in the equation is considered to be equal to zero.
        """
        
        self.bcs = solving._extract_bcs(bcs)
        # Check form style consistency
        self.is_linear = is_linear
        is_form_consistent(self.is_linear, self.bcs)
        self.Jp_eq_J = Jp is None

        self.u = u
        self.udot = udot
        self.tspan = tspan
        self.F = F
        self.G = G
        self.Jp = Jp
        if not isinstance(self.u, function.Function):
            raise TypeError(
                "Provided solution is a '%s', not a Function" % type(self.u).__name__
            )
        if not isinstance(self.udot, function.Function):
            raise TypeError(
                "Provided time derivative is a '%s', not a Function"
                % type(self.udot).__name__
            )

        # current value of time that may be used in weak form
        self.time = time or dl.Constant(0.0)
        # timeshift value provided by the solver
        self.shift = Constant(1.0)

        # Use the user-provided Jacobian. If none is provided, derive
        # the Jacobian from the residual.
        self.J = self.shift * ufl_expr.derivative(F, udot) + (J or ufl_expr.derivative(F, u))

        # Derive the Jacobian for the G residual
        self.dGdu = ufl_expr.derivative(G, u) if G is not None else None

        # Argument checking
        check_pde_args(self.F, self.G, self.J, self.Jp)

        # Store form compiler parameters
        self.form_compiler_parameters = form_compiler_parameters
        self._constant_jacobian = False
        self._constant_rhs_jacobian = False

    def dirichlet_bcs(self):
        for bc in self.bcs:
            yield from bc.dirichlet_bcs()

    def dm(self):
        return self.u.function_space().dm
'''

class TS_VariationalProblem(object):
    def __init__(self, F:ufl.Form, udot:dl.Function, u:dl.Function, bcs:list, G:ufl.Form=None):
        self.u = u  # initial iterate, state
        self.V = self.u.function_space()
        du = dl.TrialFunction(self.V)
        
        self.udot = udot  # initial iterate, time derivative
        self.Vdot = self.udot.function_space()
        dudot = dl.TrialFunction(self.Vdot)
        
        self.L = F  # residual form
        self.shift = dl.Constant(1.0)  # to be updated by the solver
        self.J_form = self.shift * dl.derivative(F, udot) + dl.derivative(F, u)  # Jacobian form
        
        # Derive the Jacobian for the G residual
        self.dGdu_form = dl.derivative(G, u) if G is not None else None  # Jacobian for G residual
        
        self.bcs = bcs
        
    def evalFunction(self, ts, t, x, xdot, f):
        # wrap PETSc vectors around dolfin functions
        x = dl.PETScVector(x)
        xdot = dl.PETScVector(xdot)
        Fvec = dl.PETScVector(f)
        
        # copy PETSc iterate to dolfin
        x.vec().copy(self.u.vector().vec())     # copy PETSc iterate to dolfin
        self.u.vector().apply("")               # update ghost values
        
        xdot.vec().copy(self.udot.vector().vec())       # copy PETSc iterate to dolfin
        self.udot.vector().apply("")                    # update ghost values
        
        dl.assemble(self.L, tensor=Fvec)        # assemble residual
        
        # apply boundary conditions.
        for bc in self.bcs:
            bc.apply(Fvec, x)
            bc.apply(Fvec, self.u.vector())
            
    
    def evalJacobian(self, ts, t, x, xdot, shift, J, P):
        self.shift = shift
        
        J = dl.PETScMatrix(J)
        x.copy(self.u.vector().vec())       # copy PETSc iterate to dolfin
        self.u.vector().apply("")           # update ghost values
        
        xdot.copy(self.udot.vector().vec())     # copy PETSc iterate to dolfin
        self.udot.vector().apply("")            # update ghost values
        dl.assemble(self.J_form, tensor=J)
        
        # apply boundary conditions.
        for bc in self.bcs:
            bc.apply(J)
            # bc.apply(P)  # todo: handle preconditioner


# class TS_VariationalSolver():
#     def __init__(self, problem, comm):
#         self.problem = problem
#         self.comm = comm
        
#         self.ts = PETSc.TS().create(comm=comm)

'''
if problem.G is None:
            # If G is not provided set the equation type as implicit
            # leave a default type otherwise
            self.ts.setEquationType(PETSc.TS.EquationType.IMPLICIT)
        else:
            # If G is provided use the arkimex solver
            self.set_default_parameter("ts_type", "arkimex")
'''
        
        
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"I'm rank {rank} of {size}", flush=True)


t0 = 0.         # simulation start time
tf = 2.         # simulation end time
num_steps = 20  # number of time steps
alpha = 3       # simulation parameter
beta = 1.2      # simulation parameter

nx = 5          # number of mesh points in x-direction
ny = 5          # number of mesh points in y-direction

domain = dl.UnitSquareMesh(comm, 10, 10)


# TODO
V = dl.FunctionSpace(mesh, "CG", 1)

u = dl.Function(V)
u_t = dl.Function(V)
v = dl.TestFunction(V)
F = dl.inner(u_t, v) * dl.dx + dl.inner(dl.grad(u), dl.grad(v)) * dl.dx - 1.0 * v * dl.dx

u0_expr = dl.Expression("x[0]*(1.-x[0])*x[1]*(1.-x[1])", element=V.ufl_element())
u0 = dl.interpolate(u0_expr, V)

# Define the boundary condition
u_D = dl.Constant(0.0)
bc = dl.DirichletBC(V, u_D, "on_boundary")





prob = TS_VariationalProblem(F, u_t, u0, [bc])

ts = PETSc.TS().create(comm=comm)
ts.setType(PETSc.TS.Type.BEULER)
ts.setProblemType(PETSc.TS.ProblemType.LINEAR)  # todo: needed for linear problems?

# vec = dl.PETScVector()
# mat = dl.PETScMatrix()

# you need to pass objects of the correct size
vec = u0.vector().vec().duplicate()
M = dl.assemble( dl.inner(dl.TrialFunction(V), dl.TestFunction(V)) * dl.dx )
Ap = dl.as_backend_type(M).mat()

ts.setIFunction(prob.evalFunction, vec)
ts.setIJacobian(prob.evalJacobian, Ap)
print('set funcs')

ts.setTime(0.0)
import numpy as np
ts.setTimeSpan(np.linspace(t0, tf, 21))
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
print('set opts')

ts.setFromOptions()             # Apply run-time options, e.g. -ts_adapt_monitor -ts_type arkimex -snes_converged_reason
ts.setSaveTrajectory()
# ts.view()  # segfault
# breakpoint()
# read from CLI / .petscrc

ts.solve(prob.u.vector().vec())
print('oh mama, we solved it')

# breakpoint()
