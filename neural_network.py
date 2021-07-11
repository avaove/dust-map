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
NUM_EPOCHS = 3
NUM_HIDNEURONS = 100
STEPS_PER_EPOCH = NUM_TRAIN//BATCH_SIZE
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.01, #initial learning rate
  decay_steps=STEPS_PER_EPOCH*20, 
  decay_rate=1, 
  staircase=False)

# >FIXME don't know if this works yet
def plot_NN_loss(train_loss, val_loss, trainLossLabel='loss', valLossLabel='val_loss', title = 'Training vs Validation Loss', color='blue'):
    '''Plot val and train loss over epochs'''
    plt.plot(train_loss, label=trainLossLabel, color=color, linestyle='-', linewidth = 1, marker = 'o', ms = 1, markeredgecolor='black', markeredgewidth=0.2)
    plt.plot(val_loss, label=valLossLabel,color=color, linestyle='dashed', linewidth = 1, ms = 1, markeredgecolor='black', markeredgewidth=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title(title)

def loss_fn(y_true, y_pred, train=False, valid=False):
    '''Return loss of data with error bars in both Y and X
    y_true has shape (BATCH_SIZE) its a batch either from Yo_train or Yo_valid
    y_pred has shape (BATCH_SIZE * 10), prediction of model when given a batch of X
    set train or valid parameter to true based on which set the loss function is given'''
    # to resolve type mismatch
    y_true = tf.cast(y_true, tf.double)
    y_pred = tf.cast(y_pred, tf.double)
    # reshape y_pred to (BATCH_SIZE, 10)
    y_pred = tf.reshape(y_pred, [BATCH_SIZE, 10])
    loss = 0
    Ye = Ye_train if train else Ye_valid # set errors to Ye_train or Ye_valid based on flag 
    for i in range(BATCH_SIZE):
        for j in range(10):
            diff = tf.math.subtract(y_true[i], y_pred[i][j])
            diff = tf.cast(diff, tf.float64)
            loss += tf.math.divide(tf.square(diff), np.square(Ye[i]))
    return loss / (BATCH_SIZE * 10)

# @tf.function
def train_step(x_batch_train, y_batch_train, model):
    # open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    with tf.GradientTape() as tape:
        # give model() x_batch_train reshaped to (BATCH_SIZE * 10, 2) so model can make logits
        x_batch_train = tf.reshape(x_batch_train, [BATCH_SIZE * 10, 2])
        # give loss_fn the (BATCHSIZE * 10) outputs from model() and find loss with BATCHSIZE Y
        logits = model(x_batch_train, training=True)  # logits for this minibatch
        loss_value = loss_fn(y_batch_train, logits) # loss value for this minibatch  
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

# @tf.function
def val_step(x_batch_valid, y_batch_valid, model):
    x_batch_valid = tf.reshape(x_batch_valid, [BATCH_SIZE * 10, 2])
    val_logits = model(x_batch_valid, training=False)
    loss_value = loss_fn(y_batch_valid, val_logits) 
    return loss_value

# @tf.function
def get_test_pred(model):
    Xo_samp_test_flattened = tf.reshape(Xo_samp_test, [NUM_TEST * 10, 2])
    pred = model(Xo_samp_test_flattened, training=False)
    pred = pred.reshape(pred, [NUM_TEST, 10, 2])
    return pred

def get_NN_model():
    '''https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
    return vals and train loss '''
    # prepare training and validation sets
    # shuffle validation and training indecies
    train_ind, valid_ind = [i for i in range(NUM_TRAIN)], [i for i in range(NUM_VALID)] # to get same shuffle indecies for X and Y
    random.shuffle(train_ind)
    random.shuffle(valid_ind)
    Xo_train_batched = tf.data.Dataset.from_tensor_slices([Xo_samp_train[i] for i in train_ind]).batch(BATCH_SIZE)
    Yo_train_batched = tf.data.Dataset.from_tensor_slices([Yo_train[i] for i in train_ind]).batch(BATCH_SIZE)
    Xo_valid_batched = tf.data.Dataset.from_tensor_slices([Xo_samp_valid[i] for i in valid_ind]).batch(BATCH_SIZE)
    Yo_valid_batched = tf.data.Dataset.from_tensor_slices([Yo_valid[i] for i in valid_ind]).batch(BATCH_SIZE)
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
            loss_value = train_step(x_batch_train, y_batch_train, model)
            train_loss.append(loss_value)
            # log every 10 batches since for train we have 60 batches and for valid we have 20 batches
            if (step % 10 == 0):
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))
        # Run a validation loop at the end of each epoch.
        for (x_batch_valid, y_batch_valid) in zip(Xo_valid_batched, Yo_valid_batched):
            loss_value = val_step(x_batch_valid, y_batch_valid, model)
            val_loss.append(loss_value) # save val loss 
    # >FIXME add defn for test_pred
    # test_pred = get_test_pred(model)
    return train_loss, val_loss #, test_pred

