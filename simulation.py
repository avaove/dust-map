from imports import *

# initialize random number seed, so that results are replicable
random_seed = 7875  # this is "SURP" on the number pad (can be changed)
np.random.seed(random_seed)

# import GP regressor and associated kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# define our Gaussian Process
sigma, scale = 1., 1.  # set the sigma (variance) and lambda (shape parameter) terms
kernel = ConstantKernel(constant_value=sigma**2) * RBF(length_scale=scale)
gp = GaussianProcessRegressor(kernel=kernel)

# define x and y grid
xgrid, ygrid = np.linspace(-5, 5, 50), np.linspace(-5, 5, 50)

# merge grid into 2-D inputs (features)
Xgrid = np.array([[x, y] for x in xgrid for y in ygrid])

# evaluate the mean and the covariance for all the points on the grid
Ygrid_mean, Ygrid_cov = gp.predict(Xgrid, return_cov=True)

# generate a realization of the log-GP at location of the inputs
# set random_state to a number to get deterministic results
Ygrid_samps = gp.sample_y(Xgrid, random_state=2021).reshape(len(Xgrid))

from scipy import interpolate
logdust = interpolate.Rbf(Xgrid[:, 0], Xgrid[:, 1], Ygrid_samps, 
                          function='thin_plate',  # specific interpolation method
                          smooth=0)  # smoothing parameter (0=exact fit)
print('finished running simulation')