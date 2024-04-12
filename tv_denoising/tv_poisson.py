import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
import time

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_DEV_PATH') )
import hippylib as hp

from poisson import PoissonBox

## constants
DO_LCURVE = True
TVONLY = [True, False, True]
N = 64  # assumed to be the same in x, y
NOISE_LEVEL = 0.02
ALPHA = 1e-2
BETA = 1e-4
PEPS = 0.5*ALPHA  # mass matrix scaled with TV
MAX_ITER = 100
CG_MAX_ITER = 75

poisson = PoissonBox(N)
poisson.setupMesh()
poisson.setupFunctionSpaces()
poisson.setupPDE()
poisson.setupTrueParameter()
poisson.generateObservations(NOISE_LEVEL)
poisson.setupMisfit()

# set up the function spaces for the TV prior
Vhm = poisson.Vh[hp.PARAMETER]
Vhw = dl.VectorFunctionSpace(poisson.mesh, 'DG', 0)
Vhwnorm = dl.FunctionSpace(poisson.mesh, 'DG', 0)

if DO_LCURVE:
    ALPHAS = np.logspace(-1, -5, num=5, base=10)
    for _, alpha in enumerate(ALPHAS):
        print(f"Running with alpha:\t {alpha:.2e}")
    
    
    nsprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA, peps=PEPS)

nsprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA)

# set up the model describing the inverse problem
model = hp.ModelNS(poisson.pde, poisson.misfit, None, nsprior, which=TVONLY)

# set up the solver
solver_params = hp.ReducedSpacePDNewtonCG_ParameterList()
solver_params["max_iter"] = MAX_ITER
solver_params["cg_max_iter"] = CG_MAX_ITER
solver = hp.ReducedSpacePDNewtonCG(model, parameters=solver_params)

# initial guess (zero)
m = poisson.pde.generate_parameter()
m.zero()

breakpoint()

# solve the system
start = time.perf_counter()
x = solver.solve([None, m, None, None])
print(f"Time to solve: {(time.perf_counter()-start)/60:.2f} minutes")
print(f"Solver convergence criterion:\t{solver.termination_reasons[solver.reason]}")

# extract the solution and plot it
xfunname = ["state", "parameter", "adjoint"]
xfun = [hp.vector2Function(x[i], poisson.Vh[i], name=xfunname[i]) for i in range(len(poisson.Vh))]

plt.figure()
dl.plot(xfun[hp.PARAMETER])
plt.show()
