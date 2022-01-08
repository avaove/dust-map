# standard imports
from matplotlib import colors
from matplotlib import rcParams
import random as rn
from tensorflow.keras.optimizers import SGD
import pandas as pd
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from scipy import interpolate

# keras imports
np.random.seed(1337)
np.set_printoptions(precision=6, suppress=True)
# reproduce results: https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
tf.random.set_seed(2021)
tf.config.run_functions_eagerly(True)
