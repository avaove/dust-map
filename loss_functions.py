from loading_data import *

# no error_bar
def loss_lin_np(y_true, y_pred):   
    '''Return ||f(Xgrid) - Pgrid||**2'''   
    return np.sum(np.square(np.subtract(y_true, y_pred)))

def loss_asinh_np(y_true, y_pred):
    '''Return ||arcsinh(f(Xgrid)) - arcsinh(Pgrid)||**2'''
    return np.sum(np.square(np.subtract(np.arcsinh(y_pred), np.arcsinh(y_true))))

def loss_lin_tf(y_true, y_pred):
    '''Return ||f(Xgrid) - Pgrid||**2'''
    return tf.square(y_true - y_pred)

def loss_asinh_tf(y_actual, y_predicted):
    '''Return ||arcsinh(f(Xgrid)) - arcsinh(Pgrid)||**2'''
    return tf.square(tf.math.asinh(y_predicted) - tf.math.asinh(y_actual))

# with error bars
def get_Ye(Y):
    '''Return appropiate error_bars array'''
    if len(Y) == NUM_TRAIN:
        return Ye_train
    elif len(Y) == NUM_VALID:
        return Ye_valid
    return Ye_test

def get_Xo_and_Xe(Y):
    '''Return appropiate Xe and Xo based on Y array given
    ex. if Y was Yo_train then return Xo_train, Xe_train'''
    if len(X) == NUM_TRAIN:
        return Xo_train, Xe_train
    elif len(X) == NUM_VALID:
        return Xo_valid, Xe_valid
    return Xo_test, Xe_test

def loss_lin_er_np(y_true, y_pred): 
    '''Return ||f(Xgrid) - Pgrid||**2 / error_bar^2'''
    loss = 0
    sigmasY = get_Ye(y_true)
    # y_pred has shape (NUM_TRAIN, 10), (NUM_TEST, 10), or (NUM_VALID, 10) based on if we deal with training/testing/validation
    for i in range(len(y_pred)):  
        for j in range(10): 
            loss += np.divide(np.square(np.subtract(y_true[i], y_pred[i][j])), np.square(sigmasY[i]))
    return loss / len(y_true)

def loss_asinh_er_np(y_true, y_pred):
    '''Return ||arcsinh(f(Xgrid)) - arcsinh(Pgrid)||**2 / error_bar^2'''
    loss = 0
    sigmasY = get_Ye(y_true)
    for i in range(len(y_pred)):
        for j in range(10): 
            loss += np.divide(np.square(np.subtract(np.arcsinh(y_true[i]), np.arcsinh(y_pred[i][j]))), np.square(sigmasY[i]))
    return loss / len(y_true)

def loss_lin_er_np(y_true, y_pred): 
    '''Return ||f(Xgrid) - Pgrid||**2 / error_bar^2'''
    loss = 0
    sigmasY = get_Ye(y_true)
    for i in range(len(y_pred)):
        for j in range(10): 
            loss += np.divide(np.square(np.subtract(y_true[i], y_pred[i][j])), np.square(sigmasY[i]))
    return loss / len(y_true)

def loss_asinh_er_np(y_true, y_pred):
    '''Return ||arcsinh(f(Xgrid)) - arcsinh(Pgrid)||**2 / error_bar^2'''
    loss = 0
    sigmasY = get_Ye(y_true)
    for i in range(len(y_pred)):
        for j in range(10):
            loss += np.divide(np.square(np.subtract(np.arcsinh(y_true[i]), np.arcsinh(y_pred[i][j]))), np.square(sigmasY[i]))
    return loss / len(y_true)

# loss functions for NN where errors are included, are located in the neural_network.py