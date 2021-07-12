from imports import *
from loading_data import *
from fixed_bins import get_bin_pred
from neural_network import get_NN_pred
from interpolation import get_interpolation_pred

def get_pred(Xo_data, model_type, model):
    '''Return predictions of general model when given Xo_data by calling appropiate get_pred'''
    pred = []
    if model_type == 'NN':
        return get_NN_pred(model, Xo_data)
    elif model_type == 'interpolate':
        return get_interpolation_pred(model, Xo_data)
    elif model_type == 'bin':
        return get_bin_pred(model, Xo_data)
    return np.array(pred)

def plot_pred_vs_true(Xo_data, Yo_data, model_type, model, title = 'Intrinsic vs predicted logdust', color = 'blue'):
    '''Plot of NN predictions (blue) and real dust value (green)
    model_type: 'NN' or 'interpolation or 'bin
    model: NN model, interpolation function, or bin avg matrix'''
    pred = get_pred(Xo_data, model_type, model) # returns nested list. Each sublist is 10 versions of observed X
    pred = [np.average(x_obs) for x_obs in pred] # get average of 10 samples for each observed pos AND avg over 10 sample predictions
    plt.scatter(Yo_data, pred, s=1, color=color, linestyle='-', linewidth = 0.1, marker = 'D', edgecolor='black') #Yo_test are the actual dust densities of the test x,y points
    plt.xlabel('c logdust')
    plt.ylabel('Predicted logdust')
    lims = [min(np.amin(Yo_data), np.amin(pred)), max(np.amax(Yo_data), np.amax(pred))]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.title(title)

# >FIXME change function so colour map is between some values: interpolation panels look weird
def plot_map_panels(Xo_data, Yo_data, model_type, model):
    '''Creates 3 panel plot showing truth on left, NN in middle and residual on right
    samp is true if X_dataset is an array of samples'''
    pred = get_pred(Xo_data, model_type, model)
    # get average of 10 samples for each observed pos AND avg over 10 sample predictions
    pred = [np.average(x_obs) for x_obs in pred] 
    Xo_data = [np.array([np.average(x_samps[:,0]),np.average(x_samps[:,1])]) for x_samps in Xo_data]
    Xo_data = np.array(Xo_data)
    plt.figure(figsize=(20, 5))
    # start plotting
    xlim, ylim = [X_MIN, X_MAX], [Y_MIN, Y_MAX]
    plt.subplot(1, 3, 1)
    plt.scatter(Xo_data[:, 0], Xo_data[:, 1], c=Yo_data, cmap='coolwarm', s=3)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.colorbar(label=r'$\log \rho(x, y)$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title('Testing set: Actual')
    plt.subplot(1, 3, 2)
    plt.scatter(Xo_data[:, 0], Xo_data[:, 1], c=pred, 
                cmap='coolwarm', s=3)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.colorbar(label=r'$\log \rho(x, y)$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title('Testing set: Predicted')
    plt.subplot(1, 3, 3) # shifting colormap    
    plt.scatter(np.append(Xo_data[:, 0], [-5, -5]), np.append(Xo_data[:, 1], [-5, -5]), 
                c=np.append(pred - Yo_data, [-(np.amax(np.abs(Yo_data - pred))), np.amax(np.abs(Yo_data - pred))]), 
                cmap='coolwarm', s=3) #tab20
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.colorbar(label='Residual')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title('Testing set: Actual vs Predicted Residual')
    plt.tight_layout()

def plot_noisy_vs_real_data():
    '''Creates 2 panel plot showing intrinsic data on left and noisy data on the right'''
    plt.figure(figsize=(16, 6))
    xlim, ylim = [X_MIN, X_MAX], [Y_MIN, Y_MAX]
    # intrinsic panel
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap='coolwarm', s=2)
    plt.scatter(X_valid[:, 0], X_valid[:, 1], c=Y_valid, cmap='coolwarm', s=2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_valid, cmap='coolwarm', s=2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.colorbar(label=r'$\log \rho(x, y)$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title('Data simulation with no noise')
    # noisy panel
    plt.subplot(1, 2, 2)
    plt.scatter(Xo_train[:, 0], Xo_train[:, 1], c=Yo_train, cmap='coolwarm', s=2)
    plt.scatter(Xo_valid[:, 0], Xo_valid[:, 1], c=Yo_valid, cmap='coolwarm', s=2)
    plt.scatter(Xo_test[:, 0], Xo_test[:, 1], c=Yo_valid, cmap='coolwarm', s=2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.colorbar(label=r'$\log \rho(x, y)$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title('Data simulation with noise in both X and Y')
    plt.tight_layout()
    plt.savefig('intrinsic.png')
