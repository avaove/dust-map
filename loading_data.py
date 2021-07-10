from imports import *

# load simulated data
# for now: Xo and Yo are the same as X and Y since we assume we incorporate the errors
n_data = 10000
SAMP = True #indicates we are including Xo errors (using samples), set to False otherwise
data = np.load('logdust_noisy.npz')
Xgrid, logdust_grid = data['Xgrid'], data['logdust_grid']
X_train, X_valid, X_test = data['X_train'], data['X_valid'], data['X_test']
Xo_train, Xo_valid, Xo_test = data['Xo_train'], data['Xo_valid'], data['Xo_test']
if SAMP:
    Xo_samp_train, Xo_samp_valid, Xo_samp_test = data['Xo_samp_train'], data['Xo_samp_valid'], data['Xo_samp_test']
Xe_train, Xe_valid, Xe_test = data['Xe_train'], data['Xe_valid'], data['Xe_test']
Y_train, Y_valid, Y_test = data['Y_train'], data['Y_valid'], data['Y_test']
Yo_train, Yo_valid, Yo_test = data['Yo_train'], data['Yo_valid'], data['Yo_test']
Ye_train, Ye_valid, Ye_test = data['Ye_train'], data['Ye_valid'], data['Ye_test']
# min/max x and y
X_MIN, Y_MIN = -5, -5
X_MAX, Y_MAX = 5, 5
# to account for datapoints falling outside of the X_MIN, X_MAX range we have a bigger range
#(this is based on the errors in X, the larger the higher the increment)
BIN_X_MIN, BIN_X_MAX = X_MIN - 2, X_MAX + 2
BIN_Y_MIN, BIN_Y_MAX = Y_MIN - 2, Y_MAX + 2
NUM_TRAIN, NUM_TEST, NUM_VALID = 6000, 2000, 2000