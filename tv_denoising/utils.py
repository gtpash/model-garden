from typing import Tuple
import sys
import os

import dolfin as dl
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR') )
import hippylib as hp

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

def generateNoisyPointwiseObservations(pde:hp.PDEProblem, B:hp.PointwiseStateObservation, m:dl.Vector, noise_level:float=0.02)->Tuple[dl.Vector, float]:
    """Add noise to pointwise observations.

    Args:
        :code:`pde`: :code:`hIPPYlib` PDE problem.
        :code:`B`: :code:`hIPPYlib` pointwise observation operator.
        :code:`m`: :code:`dolfin` vector representing the parameter.
        :code:`noise_level`: noise level.

    Returns:
        :code:`dl.Vector`: noisy pointwise observation.
        :code:`float`: noise level.
    """
    # generate state, solve the forward problem
    utrue = pde.generate_state()
    x = [utrue, m, None]
    pde.solveFwd(x[hp.STATE], x)
    
    # apply observation operator, determine noise
    breakpoint()
    data = B*x[hp.STATE]
    MAX = data.norm("linf")
    noise_std_dev = noise_level * MAX
    
    noise = dl.Vector(data).zero()
    hp.parRandom.normal(noise_std_dev, noise)
    
    # TODO: check this
    # add noise to measurements
    return data.axpy(1., noise), noise_std_dev
