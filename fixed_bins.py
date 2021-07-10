from imports import *
from loading_data import *
from general_plotting_and_model_prediction import *

def get_bin_model(n_bins):
    '''Return bin values matrix'''
    n, xbins, ybins = np.histogram2d(Xo_train[:, 0], Xo_train[:, 1],
                                    bins = [np.linspace(BIN_X_MIN, BIN_X_MAX, n_bins + 1), 
                                            np.linspace(BIN_Y_MIN, BIN_Y_MAX, n_bins + 1)])
    s, xbins, ybins = np.histogram2d(Xo_train[:, 0], Xo_train[:, 1], weights = Yo_train,
                                    bins = [np.linspace(BIN_X_MIN, BIN_X_MAX, n_bins + 1), 
                                            np.linspace(BIN_Y_MIN, BIN_Y_MAX, n_bins + 1)])
    n = np.where(n == 0, np.nan, n) # to replace 0s with nans: got error for 0/0 (weird)
    avg_mat = s / n

    while np.any(np.argwhere(np.isnan(avg_mat))): # keep going until no nan bins
        for i, j in np.argwhere(np.isnan(avg_mat)): #loop over nan indecies
            # horizontal/vertical bins
            horiz_vert_ind = [ind for ind in [(i - 1, j), (i + 1, j), (i, j + 1), (i, j - 1)] if (0 <= ind[0] < n_bins and 0 <= ind[1] < n_bins)] # get valid vertical/horiz indecies
            horiz_vert_vals = [avg_mat[ind[0]][ind[1]] for ind in horiz_vert_ind if (not np.isnan(avg_mat[ind[0]][ind[1]]))] # get non-nan horiz/vertical values
            horiz_vert_avg = np.nan # all vertical/horiz values are non
            if (len(horiz_vert_vals) != 0):
                horiz_vert_avg = np.average(horiz_vert_vals) # get average if at least 1 non-nan value
            # diagonal bins
            diag_ind = [ind for ind in [(i + 1, j + 1), (i + 1, j - 1), (i - 1, j + 1), (i - 1, j - 1)] if (0 <= ind[0] < n_bins and 0 <= ind[1] < n_bins)] 
            diag_vals = [avg_mat[ind[0]][ind[1]] for ind in diag_ind if (not np.isnan(avg_mat[ind[0]][ind[1]]))]
            diag_avg = np.nan 
            if (len(diag_vals) != 0):
                diag_avg = np.average(diag_vals) 
            # replace nan bin with vertical/horiz bin avgs, if those are all non, replace with diagonal bin avgs
            if not np.isnan(horiz_vert_avg): 
                avg_mat[i][j] = horiz_vert_avg
            elif not np.isnan(diag_avg):
                avg_mat[i][j] = diag_avg
            # all directions are nan keep going
    return avg_mat

def get_bin_value_of_pos(x, y, avg_mat, bin_len):
    '''Return bin value of bin point (x,y) falls into
    X_dataset: position of data
    Y_dataset: log(dust)'''
    # to get a start from 0 instead of -5
    xo = x - BIN_X_MIN
    yo = y - BIN_Y_MIN
#     print(xo, yo, len(avg_mat))
    return avg_mat[int(xo / bin_len)][int(yo / bin_len)]

def plot_loss_bin_model(min_bin_num, max_bin_num, X_dataset, Y_dataset, error = False):
    plt.figure(figsize=(4, 4))
    '''Plot how loss changes with increasing number of bins
    Return bin models, lin loss and asinh loss values
    error is True is error measurements are taken into account'''
    bin_model_lst, loss_lin_vals, loss_asinh_vals = ([] for i in range(3)) 
    bin_nums_lst = range(min_bin_num, max_bin_num) 

    for n_bins in bin_nums_lst:
        model = get_bin_model(n_bins) # avg_mat
        bin_model_lst.append(model) 
        # test_pred has (2000, 10) shape for test dataset
        test_pred = get_pred(X_dataset, 'bin', model) #??? X_dataset changed to Xo_test
        if (not error): # no error measurements in Y
            loss_lin_vals.append(loss_lin_np(Y_dataset, test_pred))
            loss_asinh_vals.append(loss_asinh_np(Y_dataset, test_pred))
        else: # error measurements in Y
            loss_lin_vals.append(loss_lin_er_np(Y_dataset, test_pred))
            loss_asinh_vals.append(loss_asinh_er_np(Y_dataset, test_pred))

    plt.xlabel('Number of bins')
    plt.ylabel('Loss')
    plt.title('Loss over number of bins')
    plot1, = plt.plot(loss_lin_vals, color = 'red', label='loss lin', linestyle = '-', linewidth = 1, marker = 'D', ms = 2, markeredgecolor='black', markeredgewidth=0.4)
    plot2, = plt.plot(loss_asinh_vals, color = 'blue', label='loss asinh', linestyle = '-', linewidth = 1, marker = 'D', ms = 2, markeredgecolor='black',markeredgewidth=0.4)
    plt.legend()
    plt.grid(True)
    plt.savefig('loss-bins.png')
    return bin_model_lst, loss_lin_vals, loss_asinh_vals 