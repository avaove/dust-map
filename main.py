# I just pasted the imports here too to avoid going to the top and running this piece over and over, I will delete after the NN works
from imports import *
from loss_functions import *
from loading_data import *
from general_plotting import *
from fixed_bins import * 
from interpolation import *
from neural_network import * 

train_loss, val_loss = get_NN_model()
