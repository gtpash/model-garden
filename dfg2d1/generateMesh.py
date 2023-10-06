import gmsh
import meshio
import argparse
import numpy as np
from mpi4py import MPI

def gmsh2meshio(mesh, cell_type: str, prune_z=False):
    """Extract `GMSH` mesh and return `meshio` mesh.

    Args:
        mesh: GMSH mesh.
        cell_type (str): Type of mesh cells.
        prune_z (bool, optional): Remove the z-component of the mesh to return a 2D mesh. Defaults to False.

    Returns:
        out_mesh: Converted meshio mesh object.
    """
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:geometrical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:geometrical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh


def main(args):
    # mpi stuff.
    COMM = MPI.COMM_WORLD
    MODELRANK = 0
    
    # geometry.
    GDIM = 2
    L = 2.2
    H = 0.41
    CX = 0.2
    CY = 0.2
    R = 0.05
        
    # set markers for different boundaries.
    FLUID_MARKER = 1
    INLET_MARKER = 2
    OUTLET_MARKER = 3
    WALL_MARKER = 4
    OBSTACLE_MARKER = 5
    
    # build the channel geometry.
    gmsh.initialize()
    if COMM.rank == MODELRANK:
        channel = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
        obstacle = gmsh.model.occ.addDisk(CX, CY, 0, R, R)
        
        # subtract obstacle from the channel.
        fluid = gmsh.model.occ.cut([(GDIM, channel)], [(GDIM, obstacle)])
        gmsh.model.occ.synchronize()
        
        # add physical volume marker for the fluid.
        volumes = gmsh.model.getEntities(dim=GDIM)
        assert(len(volumes) == 1) # sanity check.
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], FLUID_MARKER)
        gmsh.model.setPhysicalName(volumes[0][0], FLUID_MARKER, "Fluid")
    
    # tag the different surfaces.
    inflow, outflow, walls, obstacle = [], [], [], []
    if COMM.rank == MODELRANK:
        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H/2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H/2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L/2, H, 0]) or np.allclose(center_of_mass, [L/2, 0, 0]):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])
        
        gmsh.model.addPhysicalGroup(1, walls, WALL_MARKER)
        gmsh.model.setPhysicalName(1, WALL_MARKER, "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, INLET_MARKER)
        gmsh.model.setPhysicalName(1, INLET_MARKER, "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, OUTLET_MARKER)
        gmsh.model.setPhysicalName(1, OUTLET_MARKER, "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, OBSTACLE_MARKER)
        gmsh.model.setPhysicalName(1, OBSTACLE_MARKER, "Obstacle")
        
    # Create distance field from obstacle.
    # Add threshold of mesh sizes based on the distance field
    # LcMax -                  /--------
    #                      /
    # LcMin -o---------/
    #        |         |       |
    #       Point    DistMin DistMax
    MINRESOLUTION = R / 3
    if COMM.rank == MODELRANK:
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", [obstacle[0]])
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", MINRESOLUTION)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", R)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        
    # Generate the mesh. Use second order quadrilateral elements.
    if COMM.rank == MODELRANK:
        """
        Options for meshing, quad elements are not compatible with FEniCS 2019.1.0
        
        gmsh.option.setNumber("Mesh.Algorithm", args.mesh_algorithm)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.setOrder(args.mesh_order)
        """
        gmsh.model.mesh.generate(GDIM)
        gmsh.model.mesh.optimize("Netgen")
        
        # write the output.
        gmsh.write(f"{args.mesh_file}.msh")

    # write mesh as XDMF file.
    if COMM.rank == MODELRANK:
        msh = meshio.read(f"{args.mesh_file}.msh")
        el_mesh = create_mesh(msh, "triangle", prune_z=True)
        meshio.write(f"{args.mesh_file}.xdmf", el_mesh)
        
        # write the facets.
        line_mesh = create_mesh(msh, "line")
        meshio.write(f"{args.mesh_file}_facets.xdmf", line_mesh)
        

# set up argparse.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate mesh for 2D DFG')
    parser.add_argument('--mesh_file', type=str, help='mesh file name')
    # parser.add_argument('--mesh_order', type=int, help='mesh order (default is 2)', default=1)
    # parser.add_argument('--mesh_algorithm', type=int, help='meshing algorithm (default is 2)', default=2)
    args = parser.parse_args()

    # call main script.
    main(args)