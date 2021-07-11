# standard imports
import sys
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from scipy import interpolate

# keras imports
np.random.seed(1337)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
np.set_printoptions(precision=6, suppress=True) 
from keras.optimizers import SGD
# reproduce results: https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
import random as rn
tf.random.set_seed(2021)
tf.config.run_functions_eagerly(True)

# these modify the default plotting options
# feel free to ignore these or you can play around with
# adjusting things on your own
from matplotlib import rcParams
from matplotlib import colors
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'axes.titlepad': '15.0'})
rcParams.update({'axes.labelpad': '15.0'})
rcParams.update({'font.size': 10})

np.set_printoptions(suppress=True)