import os
import argparse
import numpy as np

import mpi4py.MPI as MPI
import fenics as fe
import dolfin as dl
import ufl


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


def getBCs(W, bndrys, MARKERS, COMM):
    UMAX = 0.3
    Uin = dl.Expression(("4.0*UMAX*x[1]*(0.41 - x[1])/(0.41*0.41)", "0.0"), degree=2, UMAX=UMAX, mpi_comm=COMM)
    
    NOSLIP = (0.0, 0.0)
    
    # prepare dirichlet boundary conditions.
    bc_walls = dl.DirichletBC(W.sub(0), NOSLIP, bndrys, MARKERS["WALL_MARKER"])
    bc_cylinder = dl.DirichletBC(W.sub(0), NOSLIP, bndrys, MARKERS["OBSTACLE_MARKER"])
    bc_in = dl.DirichletBC(W.sub(0), Uin, bndrys, MARKERS["INLET_MARKER"])
    
    return [bc_cylinder, bc_walls, bc_in]


def solveSteadyNS(W, bcs, COMM, Re=20.):
    NOSLIP = (0.0, 0.0)
    
    # define variational forms.
    v, q = dl.TestFunctions(W)
    
    return w


def main(args):
    # set up MPI.
    COMM = MPI.COMM_WORLD
    MODELRANK = 0
    
    # geometry and marker constants.
    GDIM = 2
    MARKERS = buildMarkerDict()
    
    # problem specific constants.
    REYNOLDS = 20.0
    
    # read mesh, boundaries.
    mesh, bndrys = loadMesh(args, COMM)
    
    # build mixed FEM space, surface measure on cylinder.
    W, ds_cylinder = buildSpaces(mesh, bndrys, MARKERS)
    
    # prepare dirichlet boundary conditions.
    bcs = getBCs(W, bndrys, MARKERS, COMM)
    
    # solve variational problem.
    w = solveSteadyNS(W, bcs, COMM, Re=REYNOLDS)
    
    
       
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default='./mesh/dfg2d1.xdmf', help='Path to mesh file.')
    parser.add_argument('--facets', type=str, default='./mesh/dfg2d1_facets.xdmf', help='Path to mesh file with boundaries.')
    
    args = parser.parse_args()
    main(args)