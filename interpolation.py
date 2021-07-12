from imports import *
from loss_functions import *
from general_plotting_and_model_prediction import *

def get_interpolation_pred(model, Xo_data):
    '''Return predictions of model when given Xo_data 
    Xo_data can be Xo_samp_valid, Xo_samp_train, or Xo_samp_test (for a more general purpose)'''
    return np.array([model(samplelst[:, 0], samplelst[:, 1]) for samplelst in Xo_data])  

def plot_interpolate_loss(Xo_data, Yo_data):
    '''Plot loss vs smooth metric (with logarithmic scale)'''
    lin_losses,asinh_losses,costasinh_loss_vals = ([] for i in range(3))
    smooth_vals = np.logspace(-3, 5, 10) 
    for smooth in smooth_vals:
        logdust = interpolate.Rbf(Xo_train[:, 0], Xo_train[:, 1], Yo_train, 
                          function='quintic',  # specific interpolation method
                          smooth=smooth)  # smoothing parameter (0=exact fit)
        pred = get_interpolation_pred(logdust, Xo_data)
        if (not SAMP):
            lin_losses.append(loss_lin_np(Yo_data, pred)) # given as nested array: 10 diff versions of each observed X
            asinh_losses.append(loss_asinh_np(Yo_data, pred))
        else:
            lin_losses.append(loss_lin_er_np(Yo_data, pred))
            asinh_losses.append(loss_asinh_er_np(Yo_data, pred))
    plt.xlabel('Smooth metric')
    plt.ylabel('Loss')
    plt.title('Loss vs Smooth metric')
    plt.loglog(smooth_vals, lin_losses, color = 'blue', label = 'loss lin', linestyle = '-', linewidth = 1, marker = 'D', ms = 2, markeredgecolor='black',markeredgewidth=0.4)
    plt.loglog(smooth_vals, asinh_losses, color = 'red', label = 'loss asinh', linestyle = '-', linewidth = 1, marker = 'D', ms = 2, markeredgecolor='black',markeredgewidth=0.4)
    plt.grid(True)
    plt.legend()
    plt.savefig('interpolate-loss.png')