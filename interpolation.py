from imports import *
from loss_functions import *
from general_plotting_and_model_prediction import *

def plot_interpolate_loss(X_dataset, Y_dataset):
    '''Plot loss vs smooth metric (with logarithmic scale)
    X_dataset: validation/testing for predictions
    Y_dataset: validation/testing to compare to'''
    lin_losses,asinh_losses,costasinh_loss_vals = ([] for i in range(3))
    smooth_vals = np.logspace(-3, 5, 10) 
    for smooth in smooth_vals:
        logdust = interpolate.Rbf(Xo_train[:, 0], Xo_train[:, 1], Yo_train, 
                          function='quintic',  # specific interpolation method
                          smooth=smooth)  # smoothing parameter (0=exact fit)
        pred = get_pred(X_dataset, 'interpolate', logdust)
        if (not SAMP):
            lin_losses.append(loss_lin_np(Y_dataset, pred)) # given as nested array: 10 diff versions of each observed X
            asinh_losses.append(loss_asinh_np(Y_dataset, pred))
        else:
            lin_losses.append(loss_lin_er_np(Y_dataset, pred))
            asinh_losses.append(loss_asinh_er_np(Y_dataset, pred))
    plt.xlabel('Smooth metric')
    plt.ylabel('Loss')
    plt.title('Loss vs Smooth metric')
    plt.loglog(smooth_vals, lin_losses, color = 'blue', label = 'loss lin', linestyle = '-', linewidth = 1, marker = 'D', ms = 2, markeredgecolor='black',markeredgewidth=0.4)
    plt.loglog(smooth_vals, asinh_losses, color = 'red', label = 'loss asinh', linestyle = '-', linewidth = 1, marker = 'D', ms = 2, markeredgecolor='black',markeredgewidth=0.4)
    plt.grid(True)
    plt.legend()
    plt.savefig('interpolate-loss.png')