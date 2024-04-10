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


class parameter2NoisyObservations:
    def __init__(self, pde:hp.PDEProblem, m:dl.Vector, noise_level:float=0.02, B:hp.PointwiseStateObservation=None):
        self.pde = pde
        self.m = m
        self.noise_level = noise_level
        self.B = B
        
        # set up vector for the data
        self.true_data = self.pde.generate_state()
        self.noisy_data = self.pde.generate_state()
        
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
            self.noisy_data.axpy(1., self.B*x[hp.STATE])
        else:
            self.noisy_data.axpy(1., x[hp.STATE])
        MAX = self.noisy_data.norm("linf")
        self.noise_std_dev = self.noise_level * MAX
        
        # generate noise
        noise = dl.Vector(self.noisy_data)
        noise.zero()
        hp.parRandom.normal(self.noise_std_dev, noise)
        
        # add noise to measurements
        self.noisy_data.axpy(1., noise)
