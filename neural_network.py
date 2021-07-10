from imports import *
from loading_data import *
from loss_functions import *

#set of activation functions
reluAct = ['ReLU', 'Linear'] # using this! works the best
tanhAct = ['tanh', 'Linear']
softplusAct = ['Softplus', 'Linear']
LR = 0.001
BATCH_SIZE = 100
NUM_EPOCHS = 100
NUM_HIDNEURONS = 100
N_TRAIN = len(Xo_train)
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.01, #initial learning rate
  decay_steps=STEPS_PER_EPOCH*20, 
  decay_rate=1, 
  staircase=False)

def plot_NN_loss(history, trainLossLabel='loss', valLossLabel='val_loss', title = 'Training vs Validation Loss', color='blue'):
    plt.plot(history.history['loss'], label=trainLossLabel, color=color, linestyle='-', linewidth = 1, marker = 'o', ms = 1, markeredgecolor='black', markeredgewidth=0.2)
    plt.plot(history.history['val_loss'], label=valLossLabel,color=color, linestyle='dashed', linewidth = 1, ms = 1, markeredgecolor='black', markeredgewidth=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title(title)

def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)

def get_NN_model(lossFunc = 'lin', opt = None, activationFunc = ['ReLU', 'Linear'],
              trainLossLabel='loss', valLossLabel='val_loss', color='blue', lim=[0.5,0.9]):
    '''Returns the history of the model initialized, compiled and fitted
    opt is limited to sgd and adam
    lossFunc is lin or asinh'''
    # set optimizer and learning rate
    if opt == 'sgd':
        opt = SGD(lr=LR)
    elif opt == 'adam':
        opt = tf.optimizers.Adam(learning_rate=LR)
    elif opt == None:
        opt = get_optimizer() # gradually decreases
    # standardize input
    normalizer = preprocessing.Normalization()
    if (not SAMP):
        normalizer.adapt(np.array(Xo_train))
    else:
        # getting Xo_samp_train reshaped to (NUM_TRAIN * 10, 2) instead of (NUM_TRAIN, 10, 2)
        Xo_samp_train_reshaped = np.array(Xo_samp_train).reshape([NUM_TRAIN * 10, 2])
        normalizer.adapt(Xo_samp_train)
    model = keras.Sequential([normalizer,
                          keras.layers.Dense(256, activation=activationFunc[0].lower()), 
                          keras.layers.Dense(1, activation = activationFunc[1].lower())]) 
    if (not SAMP and lossFunc == 'lin'):
        model.compile(loss=loss_lin_tf, optimizer = opt) 
    elif (not SAMP and lossFunc == 'asinh'):
        model.compile(loss=loss_asinh_tf, optimizer = opt) 
    elif (SAMP and lossFunc == 'lin'):
        model.compile(loss=loss_lin_er_tf, optimizer = opt) 
    elif (SAMP and lossFunc == 'asinh'):
        model.compile(loss=loss_asinh_er_tf, optimizer = opt)
    
    if (not SAMP):
        history = model.fit(
            Xo_train, Yo_train, 
            epochs=NUM_EPOCHS, 
            verbose=0,
            batch_size = BATCH_SIZE,
            validation_data = (Xo_valid, Yo_valid))
    else:
        # flatten both Yo_train and Xo_train
        # have same Yo for every 10 samples !!!
        Yo_train_reshaped = [] # make 10 copies of each Yo value to correspond to each 10 X samples
        for i in range(NUM_TRAIN):
            for j in range(10):
                Yo_train_reshaped.append(Yo_train[i])
        Yo_train_reshaped = tf.convert_to_tensor(Yo_train_reshaped)
        Yo_valid_reshaped = [] # make 10 copies of each Yo value to correspond to each 10 X samples
        for i in range(NUM_VALID):
            for j in range(10):
                Yo_valid_reshaped.append(Yo_valid[i])
        Yo_valid_reshaped = tf.convert_to_tensor(Yo_valid_reshaped)
        Xo_train_reshaped = tf.convert_to_tensor(np.array(Xo_samp_train).reshape([NUM_TRAIN * 10, 2]))
        Xo_valid_reshaped = tf.convert_to_tensor(np.array(Xo_samp_valid).reshape([NUM_VALID * 10, 2]))
        print(len(Xo_train_reshaped), len(Yo_train_reshaped))
        history = model.fit(
            Xo_samp_train_reshaped, Yo_train_reshaped, 
            epochs=NUM_EPOCHS, 
            verbose=0,
            batch_size = BATCH_SIZE,
            validation_data = (Xo_valid_reshaped, Yo_valid_reshaped))
    title = "Training and validation loss"    
    plot_NN_loss(history, trainLossLabel, valLossLabel, title, color)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    return model