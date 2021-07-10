from imports import *
from loading_data import *
from fixed_bins import get_bin_value_of_pos

# NOTE: X_dataset can be validation or testing coordinates
def get_pred(X_dataset, model_type, model):
    '''Return predictions from model
    note: X_dataset (Xo_prime) has taked observed and errors and generated many versions of the possible values based on observed value.
          Its a nested list where sublists are of length 10'''
    pred = []
    if model_type == 'NN':
        X_reshaped = X_dataset.reshape([len(X_dataset), 2]) 
        pred = np.hstack(model.predict(X_reshaped)) 
        # reshape back
        pred = pred.reshape([len(X_dataset), 10, 2])
    elif model_type == 'interpolate':
        pred = [model(samplelst[:, 0], samplelst[:, 1]) for samplelst in X_dataset]  #evaluate positions log dust densities
    elif model_type == 'bin':
        n_bins = len(model)
        pred = [[get_bin_value_of_pos(samplelst[i][0], samplelst[i][1], model, abs(BIN_X_MAX - BIN_X_MIN) / n_bins) for i in range(10)] for samplelst in X_dataset]
    return np.array(pred)

def plot_pred_vs_true(X_dataset, Y_dataset, model_type, model, title = 'Intrinsic vs predicted logdust', color = 'blue'):
    '''Plot of NN predictions (blue) and real dust value (green)
    X_dataset, Y_dataset: true values
    model_type: 'NN' or 'interpolation or 'bin
    model: NN model, interpolation function, or bin avg matrix
    provide NN model using NN_model if model is NN
    Return predictions'''
    pred = get_pred(X_dataset, model_type, model) # returns nested list. Each sublist is 10 versions of observed X
    if SAMP:
        pred = [np.average(x_obs) for x_obs in pred] # ??? get the average of many versions of observed X value 
    plt.scatter(Y_dataset, pred, s=1, color=color, linestyle='-', linewidth = 0.1, marker = 'D', edgecolor='black') #Yo_test are the actual dust densities of the test x,y points
    plt.xlabel('Intrinsic logdust')
    plt.ylabel('Predicted logdust')
    lims = [min(np.amin(Y_dataset), np.amin(pred)), max(np.amax(Y_dataset), np.amax(pred))]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.title(title)
    
def plot_map_panels(X_dataset, Y_dataset, model_type, model):
    '''Creates 3 panel plot showing truth on left, NN in middle and residual on right
    samp is true if X_dataset is an array of samples'''
    pred = get_pred(X_dataset, model_type, model)
    # get average of 10 samples for each observed pos AND avg over 10 sample predictions
    if SAMP:
        pred = [np.average(x_obs) for x_obs in pred] # get the average of 10 dust density for the 10 samples
        X_dataset = [np.array([np.average(x_samps[:,0]),np.average(x_samps[:,1])]) for x_samps in X_dataset]
        X_dataset = np.array(X_dataset)
    plt.figure(figsize=(20, 5))
    xlim, ylim = [X_MIN, X_MAX], [Y_MIN, Y_MAX]
    plt.subplot(1, 3, 1)
    plt.scatter(X_dataset[:, 0], X_dataset[:, 1], c=Y_dataset, cmap='coolwarm', s=3)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.colorbar(label=r'$\log \rho(x, y)$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title('Testing set: Actual')
    plt.subplot(1, 3, 2)
    plt.scatter(X_dataset[:, 0], X_dataset[:, 1], c=pred, 
                cmap='coolwarm', s=3)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.colorbar(label=r'$\log \rho(x, y)$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title('Testing set: Predicted')
    plt.subplot(1, 3, 3) # shifting colormap    
    plt.scatter(np.append(X_dataset[:, 0], [-5, -5]), np.append(X_dataset[:, 1], [-5, -5]), 
                c=np.append(pred - Y_dataset, [-(np.amax(np.abs(Y_dataset - pred))), np.amax(np.abs(Y_dataset - pred))]), 
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
    plt.title('Intrinsic Data Simulation (Log-Dust)')
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
    plt.title('Noisy data')
    plt.tight_layout()
    plt.savefig('intrinsic.png')
