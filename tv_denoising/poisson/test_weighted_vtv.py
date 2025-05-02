import os
import sys
import time

import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

sys.path.append( os.environ.get('HIPPYLIB_PATH') )
import hippylib as hp

from models import MultiPoissonBox, splitCircle
from utils import add_noise_to_observations, plotPointwiseObs

# constants, initializations
VERBOSE = True
TVONLY = [True, False, True]
NOISE_LEVEL = 0.02
ALPHA = 1e2  # regularization parameter, picked from L-curve
BETA = 1e-3
PEPS = 0.5  # mass matrix scaling in preconditioner
MAX_ITER = 1000
CG_MAX_ITER = 75
MAX_BACKTRACK = 25
DO_LCURVE = False
os.makedirs("figs/multi_poisson", exist_ok=True)  # ensure figure directory exists
os.makedirs("mesh", exist_ok=True)  # ensure mesh directory exists
NDIM = 64
NPOISSON = 3

# parameters for the split circle
C = [0.5, 0.5]
R = 0.4
VL = np.log(6.)
VR = np.log(2.)
VO = np.log(10.)
CLIM = [0.5, 3]  # limits for plotting

# setup the multiple poisson problem
mp = MultiPoissonBox(NPOISSON, NDIM)
mp.setupMesh()
mp.setupFunctionSpaces()
mp.setupPDE()

phys_dim = mp.pde.Vh[hp.STATE].mesh().topology().dim()  # the physical dimension of the mesh

# assign the true parameter(s)
expr = splitCircle(cx=C[0], cy=C[1], r=R, vl=VL, vr=VR, vo=VO)
p1mtrue = dl.interpolate(expr, mp.Vhm0)

expr = splitCircle(cx=C[0], cy=C[1], r=R, vl=VR, vr=VL, vo=VO)
p2mtrue = dl.interpolate(expr, mp.Vhm0)

expr = splitCircle(cx=C[0], cy=C[1], r=R, vl=VL, vr=VR, vo=VO)
p3mtrue = dl.interpolate(expr, mp.Vhm0)

mtrue = dl.Function(mp.Vh[hp.PARAMETER])
mp.assigner.assign(mtrue, [p1mtrue, p2mtrue, p3mtrue])

# solve the model forward
utrue = mp.pde.generate_state()
mp.pde.solveFwd(utrue, [utrue, mtrue.vector(), None])

##################################################
# UFL derived expressions for the gradient / hessian
##################################################

# compute with umberto idea:
import ufl
alpha_vec = ALPHA*np.array([1.0, 0.5, 0.25])
alpha_ufl = dl.Constant(alpha_vec)
alpha_mat = np.eye(mp.npde)
np.fill_diagonal(alpha_mat, alpha_vec)
alpha = dl.Constant(alpha_mat)

wvtv_form = ufl.sqrt(ufl.inner( alpha*ufl.grad(mtrue), ufl.grad(mtrue) )  + BETA)

m_test = dl.TestFunction(mp.Vh[hp.PARAMETER])
grad_form_ufl = dl.derivative(wvtv_form, mtrue, m_test)
grad_ufl = dl.assemble(grad_form_ufl*dl.dx)

hh = dl.Function(mp.Vh[hp.PARAMETER])
rng = hp.Random(seed=1)
rng.uniform(0.0, 1.0, hh.vector())

m_trial = dl.TestFunction(mp.Vh[hp.PARAMETER])
hess_form_ufl = dl.derivative(dl.derivative(wvtv_form*dl.dx, mtrue, hh), mtrue, m_trial)
H_ufl = dl.assemble(hess_form_ufl)

##################################################
# Hand derived expressions for the gradient / hessian
##################################################

# compute the gradient with the expression derived by hand
grad_form_hand = ufl.inner(alpha * ufl.grad(mtrue), ufl.grad(m_test)) / wvtv_form
grad_hand = dl.assemble(grad_form_hand*dl.dx)

grad_diff = grad_hand.copy()
grad_diff.axpy(-1.0, grad_ufl)
print("Max abs diff in gradient:", grad_diff.norm("linf"))  # >>> 0.0

# set up the non-linear diffusion tensor
i, j, k, l = ufl.indices(4)
eye_ncomp = ufl.Identity(mp.npde)
eye_ndim = ufl.Identity(mp.mesh.topology().dim())
eye = ufl.as_tensor(eye_ncomp[i, k] * eye_ndim[j, l], [i, j, k, l])

w = dl.grad(mtrue) / wvtv_form
vv = dl.grad(mtrue) / wvtv_form

# this works but is the full action
# grad_m = dl.grad(mtrue)
# grad_h = dl.grad(hh)  # mtilde
# grad_phi = dl.grad(m_trial)  # mhat

# hess_form = sum(
#     (1.0 / wvtv_form) * alpha_ufl[i] * ufl.inner(grad_h[i, :], grad_phi[i, :])
#     - (alpha_ufl[i]**2 / wvtv_form**3) * ufl.inner(grad_m[i, :], grad_h[i, :]) * ufl.inner(grad_m[i, :], grad_phi[i, :])
#     for i in range(mp.npde)
# ) + sum(
#     - (alpha_ufl[i] * alpha_ufl[j] / wvtv_form**3) * ufl.inner(grad_m[i, :], grad_h[i, :]) * ufl.inner(grad_m[j, :], grad_phi[j, :])
#     for i in range(mp.npde) for j in range(mp.npde) if i != j
# )

# Construct the weighted identity tensor for the Hessian:
# weighted_eye[i, j, k, l] = alpha[i, k] * I{j,l}
# - alpha[i, k] is the (i, k) entry of the weight matrix (typically diagonal, with per-component weights)
# - eye_ndim[j, l] is the identity in the spatial dimension (I{j, l})
# The resulting tensor has shape (ncomp, ndim, ncomp, ndim) and is used as the diagonal part of the weighted vectorial TV Hessian.
weighted_eye = ufl.as_tensor(alpha[i,k] * eye_ndim[j,l], [i,j,k,l])

# Acoeff is the weighted vectorial TV Hessian tensor:
#   A_{ijkl} = (1/|m|_ε) * [ (weighted_eye) - 0.5 * (α_i α_j) * (w ⊗ w) - 0.5 * (α_i α_j) * (w ⊗ w) ]
# where:
#   - weighted_eye is the weighted identity tensor with indices (i,j,k,l)
#   - α_i, α_j are the weights for components i and j
#   - v = grad(m) / |m|_ε is the normalized gradient
#   - w is the slack variable (normalized by the weighted TV norm)
#   - The subtraction terms represent the coupling between components in the Hessian
Acoeff = (1.0 / wvtv_form) * ( weighted_eye
    - dl.Constant(0.5) * dl.outer(alpha*vv, alpha*w)
    - dl.Constant(0.5) * dl.outer(alpha*w, alpha*vv) )

hess_action = ufl.grad(hh)[i,j]*Acoeff[i,j,k,l]*ufl.grad(m_test)[k,l] * dl.dx
H_hand = dl.assemble(hess_action)

H_diff = H_ufl.copy()
H_diff.axpy(-1.0, H_hand)
print("Max abs diff in Hessian action:", H_diff.norm("linf")) # >>> Max abs diff in Hessian: 1.8189894035458565e-12

##################################################
# Check the wVTV hIPPYlib implementation.
##################################################

Vhm = mp.Vh[hp.PARAMETER]
Vhw = dl.TensorFunctionSpace(mp.mesh, "DG", 0, shape=(NPOISSON, phys_dim))
Vhwnorm = dl.FunctionSpace(mp.mesh, "DG", 0)
wvtv_prior = hp.weightedVTVPrior(Vhm, Vhw, Vhwnorm, alpha_vec, BETA, peps=PEPS*ALPHA)

# check cost
wvtv_prior.cost(mtrue.vector())

# check gradient
grad_out = dl.Function(Vhm)
wvtv_prior.grad(mtrue.vector(),  grad_out.vector())

grad_diff = grad_ufl.copy()
grad_diff.axpy(-1.0, grad_out.vector())
print("Max abs diff in WVTV Class gradient:", grad_diff.norm("linf"))

# check hessian
hess_action = wvtv_prior.hess_action(mtrue, w, hh)
wvtv_hess = dl.assemble(hess_action)

H_diff = H_ufl.copy()
H_diff.axpy(-1.0, wvtv_hess)
print("Max abs diff in WVTV Class Hessian action:", H_diff.norm("linf"))


breakpoint()


##################################################
# Setup observation operators
##################################################

# set up observation operator for the top right corner
xx = np.linspace(0.5, 1.0, 25, endpoint=False)
xv, yv = np.meshgrid(xx, xx)
targets = np.vstack([xv.ravel(), yv.ravel()]).T
print(f"Number of observation points: {targets.shape[0]}")
B1 = hp.assemblePointwiseObservation(mp.pde.Vh[hp.STATE], targets)

# set up observation operator for the full domain
xx = np.linspace(0.02, 1.0, 50, endpoint=False)
xv, yv = np.meshgrid(xx, xx)
targets = np.vstack([xv.ravel(), yv.ravel()]).T
print(f"Number of observation points: {targets.shape[0]}")
B2 = hp.assemblePointwiseObservation(mp.Vh[hp.STATE], targets)

# set up observation operator for the bottom left corner
xx = np.linspace(0.02, 0.5, 25, endpoint=False)
xv, yv = np.meshgrid(xx, xx)
targets = np.vstack([xv.ravel(), yv.ravel()]).T
print(f"Number of observation points: {targets.shape[0]}")
B3 = hp.assemblePointwiseObservation(mp.Vh[hp.STATE], targets)

# write out the mesh and true parameters
MESHFPATH = os.path.join("mesh", "unitsquare.xdmf")
with dl.XDMFFile(MESHFPATH) as fid:
    fid.write(mp.Vh[hp.STATE].mesh())

m1, m2, m3 = mtrue.split()
plotPointwiseObs(mp.Vh, m1, B1, MESHFPATH, fpath="figs/multi_poisson/p1_mtrue.png", name="Log Parameter", clim=CLIM)
plotPointwiseObs(mp.Vh, m2, B2, MESHFPATH, fpath="figs/multi_poisson/p2_mtrue.png", name="Log Parameter", clim=CLIM)
plotPointwiseObs(mp.Vh, m3, B3, MESHFPATH, fpath="figs/multi_poisson/p1_mtrue.png", name="Log Parameter", clim=CLIM)

##################################################
# generate noisy observations, set up misfits
##################################################
obsops = [B1, B2, B3]
misfits = []
for idx, BB in enumerate(obsops):
    noisy_data, noise_std_dev = add_noise_to_observations(utrue.data[idx], NOISE_LEVEL, BB)
    misfits.append(hp.DiscreteStateObservation(B=BB, data=noisy_data, noise_variance=noise_std_dev**2))

misfit = hp.MultiStateMisfit(misfits)

##################################################
# setup V-TV prior and model
##################################################
Vhm = mp.Vh[hp.PARAMETER]
Vhw = dl.TensorFunctionSpace(mp.mesh, "DG", 0, shape=(NPOISSON, phys_dim))
Vhwnorm = dl.FunctionSpace(mp.mesh, "DG", 0)

tvprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA, peps=PEPS*ALPHA)
model = hp.ModelNS(mp.pde, misfit, None, tvprior, which=TVONLY)

