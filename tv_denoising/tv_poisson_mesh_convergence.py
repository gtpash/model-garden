import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
import time

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_DEV_PATH') )
import hippylib as hp

from tv_denoising.models import PoissonBox

## constants, initializations
VERBOSE = True
TVONLY = [True, False, True]
NOISE_LEVEL = 0.02
ALPHA = 1e-2
BETA = 1e-4
PEPS = 0.5  # mass matrix scaling in preconditioner
MAX_ITER = 250
CG_MAX_ITER = 75
MAX_BACKTRACK = 25
POWS = [4, 9]
os.makedirs("figs", exist_ok=True)  # ensure figure directory exists

# set up the solver parameters
solver_params = hp.ReducedSpacePDNewtonCG_ParameterList()
solver_params["max_iter"] = MAX_ITER
solver_params["cg_max_iter"] = CG_MAX_ITER
solver_params["LS"]["max_backtracking_iter"] = MAX_BACKTRACK

# run the mesh convergence study
DIMS = 2**np.arange(POWS[0], POWS[1])
newton_iters = np.zeros_like(DIMS)
cg_iters = np.zeros_like(DIMS)

for i, N in enumerate(DIMS):    
    # set up the poisson problem
    poisson = PoissonBox(N)
    poisson.setupMesh()
    poisson.setupFunctionSpaces()
    poisson.setupPDE()
    poisson.setupTrueParameter()
    poisson.generateObservations(NOISE_LEVEL)
    poisson.setupMisfit()
    
    # initial guess (zero)
    m0 = poisson.pde.generate_parameter()
    m0.zero()

    # set up the function spaces for the TV prior
    Vhm = poisson.Vh[hp.PARAMETER]
    Vhw = dl.VectorFunctionSpace(poisson.mesh, 'DG', 0)
    Vhwnorm = dl.FunctionSpace(poisson.mesh, 'DG', 0)
    
    nsprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA, peps=PEPS*ALPHA)
    
    # set up the model describing the inverse problem
    model = hp.ModelNS(poisson.pde, poisson.misfit, None, nsprior, which=TVONLY)
    solver = hp.ReducedSpacePDNewtonCG(model, parameters=solver_params)
    
    # solve the system
    start = time.perf_counter()
    x = solver.solve([None, m0, None, None])
    if VERBOSE:
        print(f"Time to solve: {(time.perf_counter()-start)/60:.2f} minutes")
        print(f"Solver convergence criterion:\t{solver.termination_reasons[solver.reason]}")
        print(f"Number of Newton iterations:\t{solver.it}")
        print(f"Total number of CG iterations:\t{solver.total_cg_iter}")
        
    newton_iters[i] = solver.it
    cg_iters[i] = solver.total_cg_iter

plt.figure()
plt.plot(DIMS, newton_iters, 'kx-', label="Newton Iterations")
plt.plot(DIMS, cg_iters, 'ko-', label="CG Iterations")
plt.xlabel("Mesh Size")
plt.ylabel("Iterations")
plt.legend()
plt.savefig("figs/mesh_convergence.png")
plt.close()

r = dl.Function(poisson.pde.Vh[hp.PARAMETER])
r.vector().axpy(1., x[hp.PARAMETER])
fig = plt.figure()
dl.plot(r)
# plt.colorbar(fig)
plt.savefig("figs/reconstruction.png")
plt.close()
