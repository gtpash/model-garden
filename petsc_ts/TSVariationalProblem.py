import dolfin as dl
import ufl

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


class TS_VariationalProblem_TDBC(object):
    def __init__(self, F:ufl.Form, udot:dl.Function, u:dl.Function, bcs:list, G:ufl.Form=None):
        # time-dependent bc
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
            bc.apply(Fvec)             # current residual vector
            bc.apply(x)                # current iterate vector
            bc.apply(self.u.vector())  # current solution vector
            
    
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

