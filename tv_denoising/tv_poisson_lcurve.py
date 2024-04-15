import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
import time

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_DEV_PATH') )
import hippylib as hp

from poisson import PoissonBox

## constants, initializations
DO_LCURVE = True
VERBOSE = True
TVONLY = [True, False, True]
N = 64  # assumed to be the same in x, y
NOISE_LEVEL = 0.02
ALPHA = 1e-2
BETA = 1e-4
PEPS = 0.5  # mass matrix scaling in preconditioner
MAX_ITER = 1000
CG_MAX_ITER = 75
MAX_BACKTRACK = 25
os.makedirs("figs", exist_ok=True)  # ensure figure directory exists

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

# set up the solver parameters
solver_params = hp.ReducedSpacePDNewtonCG_ParameterList()
solver_params["max_iter"] = MAX_ITER
solver_params["cg_max_iter"] = CG_MAX_ITER
solver_params["LS"]["max_backtracking_iter"] = MAX_BACKTRACK

# run the l-curve analysis
if DO_LCURVE:
    ALPHAS = np.logspace(-1, -5, num=5, base=10)
    misfits = np.zeros_like(ALPHAS)
    regs = np.zeros_like(ALPHAS)
    
    for i, alpha in enumerate(ALPHAS):
        print(f"\nRunning with alpha:\t {alpha:.2e}")
        nsprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, alpha, BETA, peps=PEPS*alpha)
        
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
        
        _, _, regs[i], misfits[i] = model.cost(x)
        
    fig, ax = plt.subplots()
    plt.loglog(misfits, regs / ALPHAS, 'x') #todo, might need to fix this
    plt.xlabel("Data Fidelity")
    plt.ylabel("TV Regularization")
    plt.title("L-Curve for Poisson TV Denoising")
    [ax.annotate(fr"$\alpha$={ALPHAS[i]:.2e}", (misfits[i], regs[i]/ALPHAS[i])) for i in range(len(ALPHAS))]
    plt.savefig("figs/tv_poisson_lcurve.png")

else:
    nsprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA, peps=PEPS*ALPHA)
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
        
    # extract the solution and plot it
    xfunname = ["state", "parameter", "adjoint"]
    xfun = [hp.vector2Function(x[i], poisson.Vh[i], name=xfunname[i]) for i in range(len(poisson.Vh))]

    plt.figure()
    dl.plot(xfun[hp.PARAMETER])
    plt.title("Reconstructed Parameter")
    plt.savefig("figs/tv_poisson_sol.png")
    plt.close()

plt.figure()
dl.plot(poisson.mtrue)
plt.title("True Parameter")
plt.savefig("figs/tv_poisson_true.png")
plt.close()
