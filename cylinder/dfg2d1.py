import argparse
import mpi4py.MPI as MPI
import dolfin as dl


def buildMarkerDict():
    INLET_MARKER = 2
    OUTLET_MARKER = 3
    WALL_MARKER = 4
    OBSTACLE_MARKER = 5
    
    return {"INLET_MARKER": INLET_MARKER, "OUTLET_MARKER": OUTLET_MARKER, "WALL_MARKER": WALL_MARKER, "OBSTACLE_MARKER": OBSTACLE_MARKER}
    

def loadMesh(args, COMM):
    # load the mesh.
    mesh = dl.Mesh(COMM)
    with dl.XDMFFile(COMM, args.mesh) as fid:
        fid.read(mesh)
    
    # load the boundary markers into a mesh value collection.
    bndrysMVC = dl.MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
    with dl.XDMFFile(COMM, args.facets) as fid:
        fid.read(bndrysMVC, "name_to_read")
    
    # write the boundary markers into a mesh function.
    bndrys = dl.MeshFunction("size_t", mesh, bndrysMVC)
    
    return mesh, bndrys


def buildSpaces(mesh, bndrys, MARKERS):
    """Prepare function spaces for the DFG problem.

    Args:
        mesh (dl.Mesh): problem mesh.
        comm: MPI communicator.
    """
        
    # build the function spaces.
    P2 = dl.VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = dl.FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = dl.MixedElement([P2, P1])
    W = dl.FunctionSpace(mesh, TH)
    
    # prepare surface measure on the cylinder.
    ds_cylinder = dl.Measure("ds", subdomain_data=bndrys, subdomain_id=MARKERS["OBSTACLE_MARKER"])
    
    return W, ds_cylinder


def getBCs(W, bndrys, MARKERS, COMM, U0=0.3):
    Uin = dl.Expression(("4.0*UMAX*x[1]*(0.41 - x[1])/(0.41*0.41)", "0.0"), degree=2, UMAX=U0, mpi_comm=COMM)
    
    NOSLIP = (0.0, 0.0)
    
    # prepare dirichlet boundary conditions.
    bc_walls = dl.DirichletBC(W.sub(0), NOSLIP, bndrys, MARKERS["WALL_MARKER"])
    bc_cylinder = dl.DirichletBC(W.sub(0), NOSLIP, bndrys, MARKERS["OBSTACLE_MARKER"])
    bc_in = dl.DirichletBC(W.sub(0), Uin, bndrys, MARKERS["INLET_MARKER"])
    
    return [bc_cylinder, bc_walls, bc_in]


def solveSteadyNS(W, nu, bcs, COMM):
    # define variational forms.
    v, q = dl.TestFunctions(W)
    
    w = dl.Function(W)
    u, p = dl.split(w)
    
    # define the residual variational form.
    F = dl.Constant(nu)*dl.inner(dl.grad(u), dl.grad(v))*dl.dx \
        + dl.dot(dl.dot(dl.grad(u), u), v)*dl.dx \
        - p*dl.div(v)*dl.dx - q*dl.div(u)*dl.dx
    
    # solve the problem.
    dl.solve(F == 0, w, bcs, solver_parameters={"newton_solver": {'linear_solver': 'mumps', "absolute_tolerance": 1e-7, "relative_tolerance": 1e-6}})
    
    return w


def computeQoIs(w, nu:float, ds_cylinder, COMM, U0=0.3, L=0.1):
    """Compute the quantities of interest for the DFG benchmark.

    Args:
        w (dl.Vector (mixed element)): Mixed element vector containing solution.
        nu (float): kinematic viscosity.
        ds_cylinder: dolfin measure for the cylinder surface.
        COMM: MPI communicator.
        U0 (float, optional): Free stream velocity. Defaults to 0.3.
        L (float, optional): Characteristic channel length. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    u, p = w.split()
    
    # report drag and lift.
    n = dl.FacetNormal(w.function_space().mesh())
    force = -p*n + nu*dl.dot(dl.grad(u), n)
    F_D = dl.assemble(-force[0]*ds_cylinder)
    F_L = dl.assemble(-force[1]*ds_cylinder)
    
    U_mean = 2/3*U0
    C_D = 2/(U_mean**2*L)*F_D
    C_L = 2/(U_mean**2*L)*F_L
    
    # report pressure differential.
    a_1 = dl.Point(0.15, 0.2)
    a_2 = dl.Point(0.25, 0.2)
    p_diff = p(a_1) - p(a_2)
    
    return C_D, C_L, p_diff


def main(args):
    # set up MPI.
    COMM = MPI.COMM_WORLD
    MODELRANK = 0
    
    # geometry and marker constants.
    GDIM = 2
    MARKERS = buildMarkerDict()
    
    # problem specific constants.
    nu = 0.001  # kinematic viscosity.
    U0 = 0.3  # free stream velocity.
    L = 0.1  # characteristic length.
    if COMM.rank == MODELRANK:
        print("Solving DFG 2D-1 Benchmark")
        print(f"Reynolds number is:\t{(2/3)*U0*L/nu}")
    
    # read mesh, boundaries.
    mesh, bndrys = loadMesh(args, COMM)
    
    # build mixed FEM space, surface measure on cylinder.
    W, ds_cylinder = buildSpaces(mesh, bndrys, MARKERS)
    
    # prepare dirichlet boundary conditions.
    bcs = getBCs(W, bndrys, MARKERS, COMM, U0=U0)
    
    # solve variational problem.
    w = solveSteadyNS(W, nu, bcs, COMM)
    
    # compute QoIs.
    C_D, C_L, p_diff = computeQoIs(w, nu, ds_cylinder, COMM, U0=U0, L=L)
    if COMM.rank == MODELRANK:
        print(f"Drag coefficient:\t{C_D}")
        print(f"Lift coefficient:\t{C_L}")
        print(f"Pressure difference:\t{p_diff}")
        print("NOTE: These values may differ from the benchmark since we are using Taylor-Hood elements (P2-P1) instead of Q2-P1 elements.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default='./mesh/dfg2d1.xdmf', help='Path to mesh file.')
    parser.add_argument('--facets', type=str, default='./mesh/dfg2d1_facets.xdmf', help='Path to mesh file with boundaries.')
    
    args = parser.parse_args()
    main(args)