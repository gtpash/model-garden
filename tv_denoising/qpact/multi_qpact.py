import sys
import os
import time

import ufl
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

from models import multi_qPACT, PACTMisfitForm, circularInclusion, rprint

sys.path.append( os.environ.get('HIPPYLIB_PATH') )
import hippylib as hp

COMM = dl.MPI.comm_world
dl.parameters['form_compiler']['quadrature_degree'] = 4  # set quadrature degree
FIG_DIR = "figs/multi_qpact"
os.makedirs(FIG_DIR, exist_ok=True)  # ensure figure directory exists

SEP = "\n"+"#"*80+"\n"  # for printing
NOISE_VARIANCE = 1e-6  # variance of the noise
GAMMA = 0.05  # BiLaplacian prior parameter
DELTA = 1.    # BiLaplacian prior parameter

# for the PD TV solver
DO_LCURVE = True
DO_VERIFY = True
VERBOSE = True
TVONLY = [True, False, True]
ALPHA = 1e0  # regularization parameter, picked from L-curve
BETA = 1e-3
PEPS = 0.5  # mass matrix scaling in preconditioner
MAX_ITER = 1000
CG_MAX_ITER = 75
MAX_BACKTRACK = 25

# qPACT parameters
MESH_FPATH = "mesh/circle.xdmf"
C = [2., 2.]            # center of the inclusion
R = 1.0                 # radius of the inclusion
mu_a_background = 0.01  # background absorption coefficient
mu_a_inclusion = 0.2    # inclusion absorption coefficient
D_background = 33       # background diffusion coefficient
D_inclusion = 0.2       # inclusion diffusion coefficient
u0 = [ dl.Expression("2*x[0] > 0", degree=1), dl.Expression("2*x[0] < 0", degree=1), dl.Expression("2*x[1] > 0", degree=1) ]  # incident fluence

##################################################
# set up the problem
##################################################
rprint(COMM, SEP)
rprint(COMM,"Set up the qPACT problem with the diffusion approximation.")
rprint(COMM, f"Results will be stored at: {FIG_DIR}")
rprint(COMM, SEP)

qpact = multi_qPACT(COMM, MESH_FPATH)
qpact.setupMesh()
qpact.setupFunctionSpaces()

u0fun = [ dl.interpolate(u0i, qpact.Vh[hp.STATE]) for u0i in u0 ]
qpact.setupPDE(u0fun)

##################################################
# setup the true parameter, generate noisy observations, setup the misfit
##################################################
rprint(COMM, SEP)
rprint(COMM, "Set up the true parameter, generate noisy observations.")
rprint(COMM, SEP)

mu_a_true_expr = circularInclusion(cx=C[0], cy=C[1], r=R, vin=np.log(mu_a_inclusion), vo=np.log(mu_a_background))
mu_a_fun_true = dl.interpolate(mu_a_true_expr, qpact.Vhm0)

D_true_epxr = circularInclusion(cx=C[0], cy=C[1], r=R, vin=np.log(D_inclusion), vo=np.log(D_background))
D_fun_true = dl.interpolate(D_true_epxr, qpact.Vhm0)

# assign the components of the parameter
m_fun_true = dl.Function(qpact.Vh[hp.PARAMETER])
qpact.assigner.assign(m_fun_true, [D_fun_true, mu_a_fun_true])  # parameters are [D, mu_a]

# write out the true parameters
hp.nb.multi1_plot([D_fun_true, mu_a_fun_true], ["Diffusion", "Absorption"], same_colorbar=False)
plt.savefig(os.path.join(FIG_DIR, "true_param.png"))
plt.close()

# generate true data
u_true = qpact.pde.generate_state()

x_true = [u_true, m_fun_true.vector(), None]
qpact.pde.solveFwd(u_true, x_true)

u_fun_true = [ hp.vector2Function(u_true.data[i], qpact.Vh[hp.STATE]) for i in range(u_true.nv) ]
state_names = [f"Illumination {i}" for i in range(u_true.nv)]

# add noise to the observations
noisy_data = []
for i in range(u_true.nv):
    noisy_data.append( dl.project(u_fun_true[i]*ufl.exp(m_fun_true.sub(1)), qpact.Vh[hp.STATE]) )
    hp.parRandom.normal_perturb(np.sqrt(NOISE_VARIANCE), noisy_data[i].vector())
    noisy_data[i].rename("data", "data")

# visualization
hp.nb.multi1_plot(u_fun_true, state_names, same_colorbar=True)
plt.savefig(os.path.join(FIG_DIR, "true_state.png"))
plt.close()

hp.nb.multi1_plot(noisy_data, state_names, same_colorbar=True)
plt.savefig(os.path.join(FIG_DIR, "noisy_data.png"))
plt.close()

# set up the misfit
misfits = []
for i in range(u_true.nv):
    misfit_form = PACTMisfitForm(noisy_data[i], dl.Constant(NOISE_VARIANCE))
    misfits.append( hp.NonGaussianContinuousMisfit(qpact.Vh, misfit_form) )

misfit = hp.MultiStateMisfit(misfits)

##################################################
# setup the Gaussian prior, solve for the MAP point
##################################################
rprint(COMM, SEP)
rprint(COMM, "Inverting for the MAP point with a Gaussian prior.")
rprint(COMM, SEP)

# interpolate the background values for the initial guess
mu_D = dl.Function(qpact.Vhm0)
mu_D.assign(dl.Constant(np.log(D_background)))
mu_mu_a = dl.Function(qpact.Vhm0)
mu_mu_a.assign(dl.Constant(np.log(mu_a_background)))

m0 = dl.Function(qpact.Vh[hp.PARAMETER])
qpact.assigner.assign(m0, [mu_D, mu_mu_a])

# set up the gaussian prior and model
gaussian_prior = hp.VectorBiLaplacianPrior(qpact.Vh[hp.PARAMETER], [GAMMA, GAMMA], [DELTA, DELTA])
model = hp.Model(qpact.pde, gaussian_prior, misfit)

##################################################
# run the model verification
hp.modelVerify(model, m0=m0.vector(), misfit_only=True) if DO_VERIFY else None
##################################################

xg = [model.generate_vector(hp.STATE), m0.vector(), model.generate_vector(hp.ADJOINT)]

# instantiate the solver and solve
parameters = hp.ReducedSpaceNewtonCG_ParameterList()
parameters["rel_tolerance"] = 1e-6
parameters["abs_tolerance"] = 1e-9
parameters["max_iter"]      = 500
parameters["cg_coarse_tolerance"] = 5e-1
parameters["globalization"] = "LS"
parameters["GN_iter"] = 20
if COMM.rank != 0:
    parameters["print_level"] = -1
    
solver = hp.ReducedSpaceNewtonCG(model, parameters)
xg = solver.solve(xg)

mg_fun = hp.vector2Function(xg[hp.PARAMETER], qpact.Vh[hp.PARAMETER], name = "m_map")
ug_fun = [ hp.vector2Function(xg[hp.STATE].data[i], qpact.Vh[hp.STATE]) for i in range(u_true.nv) ]

obs_fun = [ dl.project(ug_fun[i]*ufl.exp(mg_fun.sub(1)), qpact.Vh[hp.STATE]) for i in range(u_true.nv) ]
[ obs_fun[i].rename("obs", "obs") for i in range(u_true.nv) ]

# visualization
hp.nb.multi1_plot([mg_fun.sub(0), mg_fun.sub(1)], ["Diffusion", "Absorption"], same_colorbar=False)
plt.savefig(os.path.join(FIG_DIR, "param_reconstruction_gaussian.png"))
plt.close()

hp.nb.multi1_plot(ug_fun, state_names, same_colorbar=False)
plt.savefig(os.path.join(FIG_DIR, "state_reconstruction_gaussian.png"))
plt.close()

hp.nb.multi1_plot(obs_fun, state_names, same_colorbar=False)
plt.savefig(os.path.join(FIG_DIR, "obs_reconstruction_gaussian.png"))
plt.close()

##################################################
# setup the TV prior and primal-dual solver
##################################################
Vhw = dl.TensorFunctionSpace(qpact.mesh, "DG", 1, shape=(2, 2))
Vhwnorm = dl.FunctionSpace(qpact.mesh, "DG", 0)
tvprior = hp.TVPrior(qpact.Vh[hp.PARAMETER], Vhw, Vhwnorm, alpha=ALPHA, beta=BETA, peps=PEPS*ALPHA)

# set up the solver parameters
solver_params = hp.ReducedSpacePDNewtonCG_ParameterList()
solver_params["max_iter"] = MAX_ITER
solver_params["cg_max_iter"] = CG_MAX_ITER
solver_params["LS"]["max_backtracking_iter"] = MAX_BACKTRACK

model = hp.ModelNS(qpact.pde, misfit, None, tvprior, which=TVONLY)
solver = hp.ReducedSpacePDNewtonCG(model, parameters=solver_params)

##################################################
# perform an L-Curve analysis for the TV regularization parameter (if requested)
##################################################
if DO_LCURVE:
    rprint(COMM, SEP)
    rprint(COMM, "Running L-Curve analysis to determine TV regularization coefficient.")
    rprint(COMM, SEP)
    
    ALPHAS = np.logspace(2, -4, num=16, base=10)
    misfits = np.zeros_like(ALPHAS)
    regs = np.zeros_like(ALPHAS)
    
    for i, alpha in enumerate(ALPHAS):
        print(f"\nRunning with alpha:\t {alpha:.2e}")
        nsprior = hp.TVPrior(qpact.Vh[hp.PARAMETER], Vhw, Vhwnorm, alpha, BETA, peps=PEPS*alpha)
        
        # set up the model describing the inverse problem
        model = hp.ModelNS(qpact.pde, misfit, None, nsprior, which=TVONLY)
        solver = hp.ReducedSpacePDNewtonCG(model, parameters=solver_params)
        
        # solve the system
        start = time.perf_counter()
        x = solver.solve([None, m0.vector(), None, None])
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
    plt.title("L-Curve for qPACT TV Denoising")
    [ax.annotate(fr"$\alpha$={ALPHAS[i]:.2e}", (misfits[i], regs[i]/ALPHAS[i])) for i in range(len(ALPHAS))]
    plt.savefig(os.path.join(FIG_DIR, "tv_poisson_lcurve.png"))

##################################################
# solve the inverse problem with total variation regularization
##################################################
rprint(COMM, SEP)
rprint(COMM, "Inverting for the parameter with TV regularization.")
rprint(COMM, SEP)

xtv = solver.solve([None, m0.vector(), None, None])
mtv_fun = hp.vector2Function(xtv[hp.PARAMETER], qpact.Vh[hp.PARAMETER], name = "m_map")

utv_fun = [ hp.vector2Function(xtv[hp.STATE].data[i], qpact.Vh[hp.STATE]) for i in range(u_true.nv) ]
obstv_fun = [ dl.project(utv_fun[i]*ufl.exp(mg_fun.sub(1)), qpact.Vh[hp.STATE]) for i in range(u_true.nv) ]
[ obstv_fun[i].rename("obs", "obs") for i in range(u_true.nv) ]

# visualization
hp.nb.multi1_plot([mtv_fun.sub(0), mtv_fun.sub(1)], ["Diffusion", "Absorption"], same_colorbar=False)
plt.savefig(os.path.join(FIG_DIR, "param_reconstruction_tv.png"))
plt.close()

hp.nb.multi1_plot(utv_fun, state_names, same_colorbar=False)
plt.savefig(os.path.join(FIG_DIR, "state_reconstruction_tv.png"))
plt.close()

hp.nb.multi1_plot(obstv_fun, state_names, same_colorbar=False)
plt.savefig(os.path.join(FIG_DIR, "obs_reconstruction_tv.png"))
plt.close()
