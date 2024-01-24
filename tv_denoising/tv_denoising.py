'''
This script is an implementation of the CMIS lab code: https://github.com/uvilla/cmis_labs/blob/master/01_ImageDenoising/ImageDenoising_TV.ipynb
Used to verify the results of using total variation regularization.
'''

import numpy as np
import hippylib as hp
import dolfin as dl
import scipy.io as sio
import matplotlib.pyplot as plt
import logging

# Suppress FEniCS output
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

# Load in the data.
data = sio.loadmat("circles.mat")["im"]
Lx = 1.
h = Lx/float(data.shape[0])
Ly = float(data.shape[1])*h

mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(Lx, Ly), data.shape[0], data.shape[1])
Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
true = hp.NumpyScalarExpression2D()
true.setData(data, h, h)
m_true = dl.interpolate(true, Vh)

# Add noise to the image.
np.random.seed(1)
noise_stddev = 0.3
noise = noise_stddev*np.random.randn(*data.shape)
noisy = hp.NumpyScalarExpression2D()
noisy.setData(data + noise, h, h)
d = dl.interpolate(noisy, Vh)

# For sclaing.
vmin = np.min(d.vector().get_local())
vmax = np.max(d.vector().get_local())

plt.figure()
dl.plot(d)
plt.show()
dl.plot(m_true)
plt.show()