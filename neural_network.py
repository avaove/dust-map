from imports import *
from loss_functions import *
from loading_data import *
import random
import time

#set of activation functions
reluAct = ['ReLU', 'Linear'] # using this! works the best
tanhAct = ['tanh', 'Linear']
softplusAct = ['Softplus', 'Linear']
#constant vars
LR = 0.001
BATCH_SIZE = 10
NUM_EPOCHS = 100
NUM_HIDDNEURONS = 256
STEPS_PER_EPOCH = NUM_TRAIN//BATCH_SIZE
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.01, #initial learning rate
  decay_steps=STEPS_PER_EPOCH*20, 
  decay_rate=1, 
  staircase=False)
optimizer = tf.keras.optimizers.Adam(lr_schedule)

# >FIXME don't know if this works yet
def plot_NN_loss(train_loss, val_loss, trainLossLabel='loss', valLossLabel='val_loss', title = 'Training vs Validation Loss'):
    '''Plot val and train loss over epochs'''
    plt.plot(train_loss, label=trainLossLabel, color='blue', linestyle='-', linewidth = 1, marker = 'o', ms = 1, markeredgecolor='black', markeredgewidth=0.2)
    plt.plot(val_loss, label=valLossLabel,color='red', linestyle='dashed', linewidth = 1, ms = 1, markeredgecolor='black', markeredgewidth=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title(title)

@tf.function
def loss_fn(y_true, y_pred, train=False, lin_loss=True, error=False):
    '''Return loss for data with error bars in both X and Y if error=True or loss for data with no errors if error=False
    note: y_true has shape (BATCH_SIZE) its a batch either from Yo_train or Yo_valid
    note: y_pred has shape (BATCH_SIZE * 10), prediction of model when given a batch of X that has 10 samples per item
    set train to true or false based on which set the loss function is given, training or validation
    set lin_loss to true or false based on if loss function is linear or asinh'''
    # to resolve type mismatch
    y_true = tf.cast(y_true, tf.double)
    y_pred = tf.cast(y_pred, tf.double)
    loss = 0
    if not error: # no errors in either X or Y
        for i in range(BATCH_SIZE):
            loss += tf.square(y_true[i] - y_pred[i]) if lin_loss else tf.square(tf.math.asinh(y_pred[i]) - tf.math.asinh(y_true[i]))
        tf.cast(loss, tf.float64)
        return loss / BATCH_SIZE
    # errors in both X and Y
    y_true = tf.cast(y_true, tf.double)
    y_pred = tf.cast(y_pred, tf.double)
    # reshape y_pred to (BATCH_SIZE, 10)
    y_pred = tf.reshape(y_pred, [BATCH_SIZE, 10])
    Ye = Ye_train if train else Ye_valid # set errors to Ye_train or Ye_valid based on flag 
    for i in range(BATCH_SIZE):
        for j in range(10):
            # calculate lin or asinh difference
            diff = tf.math.subtract(y_true[i], y_pred[i][j]) if lin_loss else tf.math.subtract(tf.math.asinh(y_true[i]), tf.math.asinh(y_pred[i][j]))
            diff = tf.cast(diff, tf.float64)
            loss += tf.math.divide(tf.square(diff), np.square(Ye[i]))
    return loss / (BATCH_SIZE * 10)

@tf.function
def train_step(x_batch_train, y_batch_train, model, lin_loss=True, error=False):
    '''Return train loss for a training X and Y batch'''
    # open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation
    with tf.GradientTape() as tape:
        # give model() x_batch_train if error=False else x_batch_train reshaped to (BATCH_SIZE * 10, 2) so model can make logits
        if error:
            x_batch_train = tf.reshape(x_batch_train, [BATCH_SIZE * 10, 2]) 
        # give loss_fn BATCHSIZE outputs from model if error=False else give loss_fn (BATCHSIZE * 10) outputs from model() then find loss with BATCHSIZE Y
        logits = model(x_batch_train, training=True)  # logits for this minibatch
        loss_value = loss_fn(y_batch_train, logits, train=True, lin_loss=lin_loss, error=error) # loss value for this minibatch  
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

@tf.function
def val_step(x_batch_valid, y_batch_valid, model, lin_loss=True, error=False):
    '''Return loss loss for a validation X and Y batch'''
    if error:
        x_batch_valid = tf.reshape(x_batch_valid, [BATCH_SIZE * 10, 2]) 
    val_logits = model(x_batch_valid, training=False)
    loss_value = loss_fn(y_batch_valid, val_logits, train=False, lin_loss=lin_loss, error=error) 
    return loss_value

def get_NN_pred(model, X_data, error=False):
    '''Return predictions of model when given Xo_data 
    case error=True: X_data can be Xo_samp_valid, Xo_samp_train, or Xo_samp_test (for a more general purpose)
    case error=False: X_data can be X_valid, X_train, or X_test (for a more general purpose)'''
    if not error:
        return model(X_data, training=False)
    X_data_flattened = X_data.reshape([len(X_data) * 10, 2]) 
    pred = model(X_data_flattened, training=False)
    pred = pred.reshape([len(X_data), 10, 2])
    return pred

def get_NN_model(monotonic, error=False):
    '''Return model based on if model is monotonic or not
    note: kernel_initializer is to initialize weights which are set with mean of 2 and stddev of 1 based on Deep Lattice Network paper: https://slack-files.com/files-pri-safe/T4ATLBXB2-F027ZCKC39R/deep_lattice_networks.pdf?c=1626926313-dc849c3a6e0a6c96'''
    # fit state of preprocessing layer to data being passed
    # ie. compute mean and variance of the data and store them as the layer weights
    normalizer = preprocessing.Normalization() #preprocessing.Normalization(input_shape=[2,], dtype='double')
    normalizer.adapt([np.average(x_obs) for x_obs in Xo_samp_train]) if error else normalizer.adapt(X_train)# ASK  avg for normalizer? 
    inputs = keras.Input(shape=[2,])
    x = normalizer(inputs)
    # >FIXME make sure all weights are positive after this? use constraints class in tf
    x = layers.Dense(NUM_HIDDNEURONS, activation="relu", name="dense_1")(x) if not monotonic else layers.Dense(NUM_HIDDNEURONS, activation="relu", name="dense_1", kernel_initializer=my_init(2., 1.))(x)
    x = layers.Dense(NUM_HIDDNEURONS, activation="relu", name="dense_2")(x) if not monotonic else layers.Dense(NUM_HIDDNEURONS, activation="relu", name="dense_2", kernel_initializer=my_init(2., 1.))(x)
    x = layers.Dense(NUM_HIDDNEURONS, activation="relu", name="dense_3")(x) if not monotonic else layers.Dense(NUM_HIDDNEURONS, activation="relu", name="dense_3", kernel_initializer=my_init(2., 1.))(x)
    outputs = layers.Dense(1, name="predictions")(x) if not monotonic else layers.Dense(1, name="predictions", kernel_initializer=my_init(2., 1.))(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

class my_init(tf.keras.initializers.Initializer):
    ''' >FIXME add description'''
    def __init__(self, mean, stddev):
      self.mean = mean
      self.stddev = stddev

    def __call__(self, shape, dtype=None):
        initializers = np.random.normal(self.mean, self.stddev, size=shape) # get normalized random data with np
        # update negative values to positive value 1e-5 >FIXME make this shorter with list comprehension? 
        # note: shape of array is not 1D
        initializers_non_negative = np.zeros(shape,dtype="float32") # tf needs array to be casted to float64
        initializers_non_negative[initializers_non_negative < 0] = 1e-10
        return tf.convert_to_tensor(initializers_non_negative) # convert to tensor

    def get_config(self):  # >FIXME ( ASK ??? need this?)To support serialization
      return {'mean': self.mean, 'stddev': self.stddev}

def train_and_test_NN_model(lin_loss=True, monotonic=True, error=False):
    '''Return validation and training loss data over epochs 
    Source: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
    set lin_loss to true or false based on if loss function is linear or asinh'''
    # prepare training and validation sets
    # shuffle validation and training indecies to randomize batching for Xo and Yo
    train_ind, valid_ind = [i for i in range(NUM_TRAIN)], [i for i in range(NUM_VALID)] # to get same shuffle indecies for X and Y
    random.shuffle(train_ind)
    random.shuffle(valid_ind)
    # choose the correct dataset based on error parameter and batch
    X_train_data = Xo_samp_train if error else X_train
    Y_train_data = Yo_train if error else Y_train
    X_valid_data = Xo_samp_valid if error else X_valid
    Y_valid_data = Yo_valid if error else Y_valid
    X_train_batched = tf.data.Dataset.from_tensor_slices([X_train_data[i] for i in train_ind]).batch(BATCH_SIZE)
    Y_train_batched = tf.data.Dataset.from_tensor_slices([Y_train_data[i] for i in train_ind]).batch(BATCH_SIZE)
    X_valid_batched = tf.data.Dataset.from_tensor_slices([X_valid_data[i] for i in valid_ind]).batch(BATCH_SIZE)
    Y_valid_batched = tf.data.Dataset.from_tensor_slices([Y_valid_data[i] for i in valid_ind]).batch(BATCH_SIZE)    
    model = get_NN_model(monotonic, error=error)

    # list of val loss and train loss data for plotting
    val_loss, train_loss = [], []
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        print("\nStart of epo ch %d" % (epoch,))
        
        # iterate over batches - note: x_batch_train has shape (BATCH_SIZE * 10 * 2) and y_batch_train has shape (BATCH_SIZE)
        for step, (x_batch_train, y_batch_train) in enumerate(zip(X_train_batched, Y_train_batched)):
            loss_value = train_step(x_batch_train, y_batch_train, model, lin_loss, error=error)
            # appending was here
            # log every 10 batches - note: for training we have 60 batches and for validation we have 20 batches
            # print("STEP: ", step, len(X_train_batched), len(Y_train_batched))
            if (step % 100 == 0):
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))
        # run validation loop at the end of each epoch.
        for (x_batch_valid, y_batch_valid) in zip(X_valid_batched, Y_valid_batched):
            val_loss_value = val_step(x_batch_valid, y_batch_valid, model, lin_loss, error=error)
            # appending was here
        train_loss.append(loss_value) # save train loss for plotting
        val_loss.append(val_loss_value) # save val loss for plotting
        print("Time taken: %.2fs" % (time.time() - start_time))
    # >FIXME add defn for test_pred after fixing the speed issue
    test_pred = get_NN_pred(model, Xo_samp_test) if error else get_NN_pred(model, X_test)
    return train_loss, val_loss, test_pred

