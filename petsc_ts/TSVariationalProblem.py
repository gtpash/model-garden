# This provides a class to handle interfacing with PETSc.TS objects.
# The structure of this code is inspired by the firedrake-ts implementation:
# https://github.com/IvanYashchuk/firedrake-ts

import dolfin as dl
import ufl


def check_pde_args(F, G, J, Jp):
    if not isinstance(F, ufl.form.Form):
        raise TypeError(f"Provided residual is a '{type(F).__name__}', not a UFL Form")
    if len(F.arguments()) != 1:
        raise ValueError("Provided residual is not a linear form")
    if G is not None and not isinstance(G, ufl.Form):
        raise TypeError(f"Provided G residual is a '{type(G).__name__}', not UFL Form")
    if G is not None and len(G.arguments()) != 1:
        raise ValueError("Provided G residual is not a linear form")
    if not isinstance(J, ufl.Form):
        raise TypeError(f"Provided Jacobian is a '{type(J).__name__}', not UFL Form")
    if len(J.arguments()) != 2:
        raise ValueError("Provided Jacobian is not a bilinear form")
    if Jp is not None and not isinstance(Jp, ufl.Form):
        raise TypeError(f"Provided Jacobian is a '{type(Jp).__name__}', not UFL Form")
    if Jp is not None and len(Jp.arguments()) != 2:
        raise ValueError("Provided preconditioner is not a bilinear form")


class TS_VariationalProblem(object):
    def __init__(self, Fvarf:ufl.Form, udot:dl.Function, u:dl.Function, bcs:list, G:ufl.Form=None):
        self.u = u  # initial iterate, state
        self.V = self.u.function_space()
        self.du = dl.TrialFunction(self.V)
        
        self.udot = udot  # initial iterate, time derivative
        self.Vdot = self.udot.function_space()
        self.dudot = dl.TrialFunction(self.Vdot)
        
        # Standard __call__ signature should take (t, u, udot)
        # according to manual: https://petsc.org/main/manual/ts/
        self.Fvarf_handler = Fvarf  # residual variational form handler

        # placeholder for the residual form, jacobian form
        self.F_form = None
        self.shift = None
        self.J_form = None
        
        # todo: handle the RHS term later
        # Derive the Jacobian for the G residual
        self.dGdu_form = dl.derivative(G, u) if G is not None else None  # Jacobian for G residual
        
        self.bcs = bcs
        
    def evalFunction(self, ts, t, x, xdot, f):
        # wrap PETSc vectors with dolfin functions
        x = dl.PETScVector(x)
        xdot = dl.PETScVector(xdot)
        Fvec = dl.PETScVector(f)
        
        # copy PETSc iterate to dolfin
        x.vec().copy(self.u.vector().vec())     # copy PETSc iterate to dolfin
        self.u.vector().apply("")               # update ghost values
        
        xdot.vec().copy(self.udot.vector().vec())       # copy PETSc iterate to dolfin
        self.udot.vector().apply("")                    # update ghost values
        
        # changing Fvec changes f, so PETSc will see the changes
        self.F_form = self.Fvarf_handler(t, self.u, self.udot)  # get the residual form
        dl.assemble(self.F_form, tensor=Fvec)           # assemble residual
        
        # apply boundary conditions.
        for bc in self.bcs:
            if isinstance(bc, dl.DirichletBC):
                bc.apply(Fvec)
                bc.apply(x)
                bc.apply(self.u.vector())
            else:  # assume it was a function with the standard caller signature
                bc_handler = bc(t, self.u, self.udot)
                bc_handler.apply(Fvec)
                bc_handler.apply(x)
                bc_handler.apply(self.u.vector())
            
    
    def evalJacobian(self, ts, t, x, xdot, shift, J, P):       
        Jvec = dl.PETScMatrix(J)                # wrap PETSc matrix with dolfin
        
        # todo: I don't think these are necessary
        # x.copy(self.u.vector().vec())           # copy PETSc iterate to dolfin
        # self.u.vector().apply("")               # update ghost values
        # xdot.copy(self.udot.vector().vec())     # copy PETSc iterate to dolfin
        # self.udot.vector().apply("")            # update ghost values
        
        # compute the Jacobian
        # todo: don't recompute if the problem is linear
        self.shift = shift
        self.J_form = self.shift * dl.derivative(self.F_form, self.udot, self.dudot) \
            + dl.derivative(self.F_form, self.u, self.du)  # Jacobian form
        dl.assemble(self.J_form, tensor=Jvec)  # changing Jvec changes J, so PETSc will see the changes
        
        # apply boundary conditions.
        for bc in self.bcs:
            if isinstance(bc, dl.DirichletBC):
                bc.apply(Jvec)
                # bc.apply(P)  # todo: handle preconditioner
            else:  # assume it was a function with the standard caller signature
                bc_handler = bc(t, self.u, self.udot)
                bc_handler.apply(Jvec)
                # bc.apply(P)  # todo: handle preconditioner

# todo: variational solver class to wrap the TS object.
