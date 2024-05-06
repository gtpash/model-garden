import sys
import os
from typing import List

import numpy as np
import dolfin as dl
import pyvista as pv

sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR') )
import hippylib as hp

def rprint(comm, *args, **kwargs):
    """Print only on rank 0."""
    if comm.rank == 0:
        print(*args, **kwargs)


def samplePrior(prior):
    """Wrapper to sample from a :code:`hIPPYlib` prior.

    Args:
        :code:`prior`: :code:`hIPPYlib` prior object.

    Returns:
        :code:`dl.Vector`: noisy sample of prior.
    """
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    hp.parRandom.normal(1., noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)
    return mtrue


class parameter2NoisyObservations:
    def __init__(self, pde:hp.PDEProblem, m:dl.Vector, noise_level:float=0.02, B:hp.PointwiseStateObservation=None):
        self.pde = pde
        self.m = m
        self.noise_level = noise_level
        self.B = B
        
        # set up vector for the data
        self.true_data = self.pde.generate_state()
        
        self.noise_std_dev = None
        
    def generateNoisyObservations(self):
        # generate state, solve the forward problem
        utrue = self.pde.generate_state()
        x = [utrue, self.m, None]
        self.pde.solveFwd(x[hp.STATE], x)
        
        # store the true data
        self.true_data.axpy(1., x[hp.STATE])
        
        # apply observation operator, determine noise
        if self.B is not None:
            self.noisy_data = dl.Vector(self.B.mpi_comm())  # shape vector to match B
            self.B.init_vector(self.noisy_data, 0)          # initialize vector
            self.noisy_data.axpy(1., self.B*x[hp.STATE])
        else:
            self.noisy_data = self.pde.generate_state()
            self.noisy_data.axpy(1., x[hp.STATE])
        
        MAX = self.noisy_data.norm("linf")
        self.noise_std_dev = self.noise_level * MAX
        
        # generate noise
        noise = dl.Vector(self.noisy_data)
        noise.zero()
        hp.parRandom.normal(self.noise_std_dev, noise)
        
        # add noise to measurements
        self.noisy_data.axpy(1., noise)


def interpolatePointwiseObsOp(Vh:dl.FunctionSpace, B:hp.PointwiseStateObservation)->List[dl.Function]:
    """Interpolate a PointwiseObservationOperator onto function space.

    Args:
        Vh (dl.FunctionSpace): Function space.
        B (hp.PointwiseStateObservation): Pointwise observation operator.

    Returns:
        (list): List of interpolated functions.
    """
    mesh = Vh.mesh()
    if mesh.geometry().dim() == 1:
        xyz_fun = [dl.Expression("x[0]", degree=1)]
    elif mesh.geometry().dim() == 2:
        xyz_fun = [dl.Expression("x[0]", degree=1), dl.Expression("x[1]", degree=1)]
    else:
        xyz_fun = [dl.Expression("x[0]", degree=1), dl.Expression("x[1]", degree=1), dl.Expression("x[2]", degree=1)]

    return [B*dl.interpolate(fun, Vh).vector() for fun in xyz_fun]


def plotPointwiseObs(Vh:dl.FunctionSpace, x:dl.Vector, B:hp.DiscreteStateObservation, meshfpath:str, fpath:str, name:str="state", clim=None):
    # todo: clean up
    pv.start_xvfb()
    
    # read in the mesh
    meshReader = pv.get_reader(meshfpath)
    grid = meshReader.read()
    
    # interpolate the pointwise observation operator
    xyz = interpolatePointwiseObsOp(Vh[hp.STATE], B)
    xyz_array = np.stack([xi.get_local() for xi in xyz])
    xyz_array = xyz_array.T  # shape (N, dim)
    
    # pad with zeros for the third dimension (if necessary)
    if xyz_array.shape[1] == 2:
        xyz_array = np.hstack([xyz_array, np.zeros((xyz_array.shape[0], 1))])
        
    xyzPoints = pv.PolyData(xyz_array)
    
    # interpolate the parameter, add to the grid
    grid[name] = x.compute_vertex_values()
    
    # set up the plotter
    p = pv.Plotter(off_screen=True, lighting=None)
    p.add_mesh(grid, style="wireframe")
    p.add_mesh(grid, scalars=name)
    
    if clim is not None:
        p.update_scalar_bar_range(clim)  # manually adjust the colorbar range
    
    p.add_mesh(xyzPoints, color="white", point_size=3, render_points_as_spheres=False)
    
    p.view_xy()
    p.screenshot(fpath)


def plotScalarFunction(Vh:dl.FunctionSpace, grid:pv.Grid, u:dl.Function, name:str="state", **kwargs)->pv.Plotter:
    """Plot a scalar function.

    Args:
        Vh (dl.FunctionSpace): Function space for variable.
        grid (pv.Grid): PyVista grid.
        u (dl.Function): Function to be plotted.

    Returns:
        (pv.Plotter): PyVista Plotter object.
    """
    # Set up the plotter.
    p = pv.Plotter(off_screen=True, lighting=None)
    p.clear()
    
    grid[name] = u.compute_vertex_values()
    
    p.add_mesh(grid, style="wireframe")
    p.add_mesh(grid, scalars=name, **kwargs)
    p.view_xy()

    return p