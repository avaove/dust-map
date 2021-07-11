from imports import *
from loss_functions import *
from loading_data import *
import random

#set of activation functions
reluAct = ['ReLU', 'Linear'] # using this! works the best
tanhAct = ['tanh', 'Linear']
softplusAct = ['Softplus', 'Linear']
LR = 0.001
BATCH_SIZE = 100
NUM_EPOCHS = 10
NUM_HIDNEURONS = 100
N_TRAIN = len(Xo_train)
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.01, #initial learning rate
  decay_steps=STEPS_PER_EPOCH*20, 
  decay_rate=1, 
  staircase=False)

# loss = 0
#     sigmasY = get_Ye(y_true)
#     for i in range(len(y_pred)):
#         for j in range(10): # loop over diff versions of 1 X observation
#             loss += np.divide(np.square(np.subtract(y_true[i], y_pred[i][j])), np.square(sigmasY[i]))
#     return loss / len(y_true)

def loss_fn(y_true, y_pred, train=False, valid=False):
    '''y_true has shape (BATCH_SIZE) its a batch either from Yo_train or Yo_valid
    y_pred has shape (BATCH_SIZE * 10), prediction of model when given a batch of X
    set train or valid parameter to true based on which set the loss function is given'''
    # to resolve type mismatch
    y_true = tf.cast(y_true, tf.double)
    y_pred = tf.cast(y_pred, tf.double)
    y_pred = tf.reshape(y_pred, [BATCH_SIZE, 10]) # reshape y_pred to (BATCH_SIZE, 10)
    loss = 0
    Ye = Ye_train if train else Ye_valid # set errors to Ye_train or Ye_valid based on flag 
    for i in range(len(y_pred)):
        for j in range(10):
            diff = tf.math.subtract(y_true[i], y_pred[i][j])
            diff = tf.cast(diff, tf.float64)
            loss += tf.math.divide(tf.square(diff), np.square(Ye[i]))
    return loss / len(y_true)

def get_NN_model():
    '''https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
    return vals '''
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    # prepare training and validation sets
    # shuffle validation and training indecies
    train_ind, valid_ind = [i for i in range(NUM_TRAIN)], [i for i in range(NUM_VALID)] # to get same shuffle indecies for X and Y
    random.shuffle(train_ind)
    random.shuffle(valid_ind)
    Xo_train_batched = tf.data.Dataset.from_tensor_slices([Xo_samp_train[i] for i in train_ind]).batch(BATCH_SIZE)
    Yo_train_batched = tf.data.Dataset.from_tensor_slices([Yo_train[i] for i in train_ind]).batch(BATCH_SIZE)
    Xo_valid_batched = tf.data.Dataset.from_tensor_slices([Xo_samp_valid[i] for i in valid_ind]).batch(BATCH_SIZE)
    Yo_valid_batched = tf.data.Dataset.from_tensor_slices([Yo_valid[i] for i in valid_ind]).batch(BATCH_SIZE)
    #<BatchDataset shapes: (None, 10, 2), types: tf.float64> <BatchDataset shapes: (None,), types: tf.float64>
    # print(len(list(Xo_train_batched.as_numpy_iterator())), list(Xo_train_batched.as_numpy_iterator()))
    
    # fit state of preprocessing layer to data being passed
    # ie. compute mean and variance of the data and store them as the layer weights
    normalizer = preprocessing.Normalization(input_shape=[2,], dtype='double')
    normalizer.adapt([np.average(x_obs) for x_obs in Xo_samp_train])  
    model = keras.Sequential([normalizer,
                              keras.layers.Dense(256, activation=reluAct[0].lower()),
                              keras.layers.Dense(256, activation=reluAct[0].lower()),
                              keras.layers.Dense(256, activation=reluAct[0].lower()),
                              keras.layers.Dense(1, activation = reluAct[1].lower())])
    val_loss, train_loss = [], []
    for epoch in range(NUM_EPOCHS):
        print("\nStart of epoch %d" % (epoch,))
        # iterate over batches - note: x_batch_train has shape (BATCH_SIZE * 10 * 2) and y_batch_train has shape (BATCH_SIZE)
        for step, (x_batch_train, y_batch_train) in enumerate(zip(Xo_train_batched, Yo_train_batched)):
            # open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation
            with tf.GradientTape() as tape:
                # >TODO give model() x_batch_train reshaped to (BATCH_SIZE * 10, 2) so model can make logits
                # print(len(x_batch_train), len(x_batch_train[0]), x_batch_train[0])
                x_batch_train = tf.reshape(x_batch_train, [BATCH_SIZE * 10, 2])
                # >TODO give loss_fn the (BATCHSIZE * 10) outputs from model() and find loss with BATCHSIZE Y
                logits = model(x_batch_train, training=True)  # logits for this minibatch
                loss_value = loss_fn(y_batch_train, logits) # loss value for this minibatch  
                train_loss.append(loss_value)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # log every 10 batches since for train we have 60 batches and for valid we have 20 batches
            if (step % 10 == 0):
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))
        # >TODO save loss for valid and testing in test
        # Run a validation loop at the end of each epoch.
        for (x_batch_valid, y_batch_valid) in zip(Xo_valid_batched, Yo_valid_batched):
            x_batch_valid = tf.reshape(x_batch_valid, [BATCH_SIZE * 10, 2])
            val_logits = model(x_batch_valid, training=False)
            loss_value = loss_fn(y_batch_valid, val_logits) 
            val_loss.append(loss_value) # save val loss 
    
    return train_loss, val_loss


# def plot_NN_loss(history, trainLossLabel='loss', valLossLabel='val_loss', title = 'Training vs Validation Loss', color='blue'):
#     plt.plot(history.history['loss'], label=trainLossLabel, color=color, linestyle='-', linewidth = 1, marker = 'o', ms = 1, markeredgecolor='black', markeredgewidth=0.2)
#     plt.plot(history.history['val_loss'], label=valLossLabel,color=color, linestyle='dashed', linewidth = 1, ms = 1, markeredgecolor='black', markeredgewidth=0.2)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.title(title)

# def get_NN_model(lossFunc = 'lin', opt = None, activationFunc = ['ReLU', 'Linear'],
#               trainLossLabel='loss', valLossLabel='val_loss', color='blue', lim=[0.5,0.9]):
#     '''Returns the history of the model initialized, compiled and fitted
#     opt is limited to sgd and adam
#     lossFunc is lin or asinh'''
#     # set optimizer and learning rate
#     if opt == 'sgd':
#         opt = SGD(lr=LR)
#     elif opt == 'adam':
#         opt = tf.optimizers.Adam(learning_rate=LR)
#     elif opt == None:
#         opt = get_optimizer() # gradually decreases
#     # standardize input
#     normalizer = preprocessing.Normalization()
#     if (not SAMP):
#         normalizer.adapt(np.array(Xo_train))
#     else:
#         # getting Xo_samp_train reshaped to (NUM_TRAIN * 10, 2) instead of (NUM_TRAIN, 10, 2)
#         Xo_samp_train_reshaped = np.array(Xo_samp_train).reshape([NUM_TRAIN * 10, 2])
#         normalizer.adapt(Xo_samp_train)
#     model = keras.Sequential([normalizer,
#                           keras.layers.Dense(256, activation=activationFunc[0].lower()),
#                           keras.layers.Dense(256, activation=activationFunc[0].lower()),
#                           keras.layers.Dense(1, activation = activationFunc[1].lower())]) 
#     if (not SAMP and lossFunc == 'lin'):
#         model.compile(loss=loss_lin_tf, optimizer = opt) 
#     elif (not SAMP and lossFunc == 'asinh'):
#         model.compile(loss=loss_asinh_tf, optimizer = opt) 
#     elif (SAMP and lossFunc == 'lin'):
#         model.compile(loss=loss_lin_er_tf, optimizer = opt) 
#     elif (SAMP and lossFunc == 'asinh'):
#         model.compile(loss=loss_asinh_er_tf, optimizer = opt)
    
#     if (not SAMP):
#         history = model.fit(
#             Xo_train, Yo_train, 
#             epochs=NUM_EPOCHS, 
#             verbose=0,
#             batch_size = BATCH_SIZE,
#             validation_data = (Xo_valid, Yo_valid))
#     else:
#         # flatten both Yo_train and Xo_train
#         # have same Yo for every 10 samples !!!
#         Yo_train_reshaped = [] # make 10 copies of each Yo value to correspond to each 10 X samples
#         for i in range(NUM_TRAIN):
#             for j in range(10):
#                 Yo_train_reshaped.append(Yo_train[i])
#         Yo_train_reshaped = tf.convert_to_tensor(Yo_train_reshaped)
#         Yo_valid_reshaped = [] # make 10 copies of each Yo value to correspond to each 10 X samples
#         for i in range(NUM_VALID):
#             for j in range(10):
#                 Yo_valid_reshaped.append(Yo_valid[i])
#         Yo_valid_reshaped = tf.convert_to_tensor(Yo_valid_reshaped)
#         Xo_train_reshaped = tf.convert_to_tensor(np.array(Xo_samp_train).reshape([NUM_TRAIN * 10, 2]))
#         Xo_valid_reshaped = tf.convert_to_tensor(np.array(Xo_samp_valid).reshape([NUM_VALID * 10, 2]))
#         print(len(Xo_train_reshaped), len(Yo_train_reshaped))
#         history = model.fit(
#             Xo_samp_train_reshaped, Yo_train_reshaped, 
#             epochs=NUM_EPOCHS, 
#             verbose=0,
#             batch_size = BATCH_SIZE,
#             validation_data = (Xo_valid_reshaped, Yo_valid_reshaped))
#     title = "Training and validation loss"    
#     plot_NN_loss(history, trainLossLabel, valLossLabel, title, color)
#     hist = pd.DataFrame(history.history)
#     hist['epoch'] = history.epoch
#     print(hist.tail())
#     return model
