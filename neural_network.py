from imports import *
from loss_functions import *
from loading_data import *
import random
import time

# set up
BATCH_SIZE = 100
EPOCHS = 50
HIDDEN_NEURONS = 256
STEPS_PER_EPOCH = NUM_TRAIN//BATCH_SIZE
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.01, #initial learning rate
  decay_steps=STEPS_PER_EPOCH*20, 
  decay_rate=1, 
  staircase=False)
optimizer = tf.keras.optimizers.Adam(lr_schedule)
delta_r = 0.01


def plot_NN_loss(train_loss, val_loss, trainLossLabel='loss', valLossLabel='val_loss', title = 'Training vs Validation Loss'):
    '''Plot val loss and train loss over epochs'''
    plt.plot(train_loss, label=trainLossLabel, color='blue', linestyle='-', linewidth = 1, 
             marker = 'o', ms = 2, markeredgecolor='black', markeredgewidth=0.2)
    plt.plot(val_loss, label=valLossLabel, color='red', linestyle='dashed', linewidth = 1, 
             marker = 'o', ms = 2, markeredgecolor='black', markeredgewidth=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title(title)


@tf.function
def loss_fn(y_true, y_pred, train=False, error=False):
    '''Return loss for data with error bars in both X and Y if error=True else return loss for data with no errors
    y_true has shape (BATCH_SIZE); its a batch either from Yo_train or Yo_valid
    y_pred has shape (BATCH_SIZE * 10); its prediction of model when given a batch of X that has 10 samples per item
    Set train to true if given a training set, else if given a set validation set, set train to false'''
    # to resolve type mismatch
    y_true = tf.cast(y_true, tf.double)
    y_pred = tf.cast(y_pred, tf.double)
    loss = 0
    if not error: # no errors in either X or Y
        for i in range(BATCH_SIZE):
            loss += tf.square(y_true[i] - y_pred[i]) 
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
            diff = tf.math.subtract(y_true[i], y_pred[i][j]) 
            diff = tf.cast(diff, tf.float64)
            loss += tf.math.divide(tf.square(diff), np.square(Ye[i]))
    return loss / (BATCH_SIZE * 10)


@tf.function
def train_step(x_batch_train, y_batch_train, model, error=False, num_input=2):
    '''Return train loss for a training X and Y batch'''
    # open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation
    with tf.GradientTape() as tape:
        # give model() x_batch_train if error=False else x_batch_train reshaped to (BATCH_SIZE * 10, num_input) so model can make logits
        if error:
            x_batch_train = tf.reshape(x_batch_train, [BATCH_SIZE * 10, num_input]) 
        # give loss_fn BATCHSIZE outputs from model if error=False else give loss_fn (BATCHSIZE * 10) outputs from model() then find loss with BATCHSIZE Y
        logits = model(x_batch_train, training=True)  # logits for this minibatch
        loss_value = loss_fn(y_batch_train, logits, train=True, error=error) # loss value for this minibatch  
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


@tf.function
def val_step(x_batch_valid, y_batch_valid, model, error=False, num_input=2):
    '''Return loss loss for a validation X and Y batch'''
    if error:
        x_batch_valid = tf.reshape(x_batch_valid, [BATCH_SIZE * 10, num_input]) 
    val_logits = model(x_batch_valid, training=False)
    loss_value = loss_fn(y_batch_valid, val_logits, train=False, error=error) 
    return loss_value


def get_NN_pred(model, X_data, error=False, num_input=2):
    '''Return predictions of model when given Xo_data 
    case error=True: X_data can be Xo_samp_valid, Xo_samp_train, or Xo_samp_test (for a more general purpose)
    case error=False: X_data can be X_valid, X_train, or X_test (for a more general purpose)'''
    if not error:
        return model(X_data, training=False)
    X_data_flattened = X_data.reshape([len(X_data) * 10, num_input]) 
    pred = model(X_data_flattened, training=False)
    pred_np = pred.numpy()
    pred_np = pred_np.reshape([len(X_data), 10])
    return pred_np


class My_Init(tf.keras.initializers.Initializer):
    '''Initializes weight tensors to be non negative and have mean and standard dev given'''
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=None):
        initializers = np.random.normal(self.mean, self.stddev, size=shape) # get normalized random data
        initializers = initializers.astype("float32")
        # keep the weights from the input layer to the 1st hidden layer for input theita the same
        # update negative values to positive value 1e-10
        initializers[initializers < 0] = 1e-10
        # set theita weights back
        if len(initializers) == 2 or len(initializers) == 3: # make sure this is the first layer of weights
            initializers[1] = np.random.normal(0., 0.05, size=256) # generate weights for theita input neuron (how they are generated by default)
        return tf.convert_to_tensor(initializers) # convert to tensor

class My_Constraint(tf.keras.constraints.Constraint):
  '''Constrains weight tensors to be non negative'''
  def __call__(self, w):
      w_np = w.numpy()
      # keep the weights from the input layer to the 1st hidden layer for input theita the same
      if len(w_np) == 2 or len(w_np) == 3: # make sure this is the first layer of weights
          theita_w_copy = np.copy(w[1]) # make copy of weights connected to theita
      # update negative values to positive value 1e-10
      w_np[w_np < 0] = 1e-10
      # set theita weights back
      if len(w_np) == 2 or len(w_np) == 3:
          w_np[1] = theita_w_copy
      return tf.convert_to_tensor(w_np)
  

def get_NN_model(monotonic, error=False, num_input=2):
    '''Return model based on if model is monotonic or not
    num_input=2 if (r, theita) is the NN input
    num_input=3 if (r, x/r, y/r) is NN input'''
    # fit state of preprocessing layer to data being passed
    # ie. compute mean and variance of the data and store them as the layer weights
    normalizer = preprocessing.Normalization() #preprocessing.Normalization(input_shape=[2,], dtype='double')
    normalizer.adapt([np.average(x_obs) for x_obs in Xo_samp_train]) if error else normalizer.adapt(X_train)# ASK  avg for normalizer? 
    inputs = keras.Input(shape=[num_input,]) 
    x = normalizer(inputs)
    x = layers.Dense(HIDDEN_NEURONS, activation="relu", name="dense_1")(x) if not monotonic else layers.Dense(HIDDEN_NEURONS, activation="relu", name="dense_1", kernel_initializer=My_Init(2., 1.), kernel_constraint=My_Constraint())(x)
    x = layers.Dense(HIDDEN_NEURONS, activation="relu", name="dense_2")(x) if not monotonic else layers.Dense(HIDDEN_NEURONS, activation="relu", name="dense_2", kernel_initializer=My_Init(2., 1.), kernel_constraint=My_Constraint())(x)
    x = layers.Dense(HIDDEN_NEURONS, activation="relu", name="dense_3")(x) if not monotonic else layers.Dense(HIDDEN_NEURONS, activation="relu", name="dense_3", kernel_initializer=My_Init(2., 1.), kernel_constraint=My_Constraint())(x)
    # activation is linear if not specified
    outputs = layers.Dense(1, name="predictions")(x) if not monotonic else layers.Dense(1, name="predictions", kernel_initializer=My_Init(2., 1.), kernel_constraint=My_Constraint())(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_NN_model(monotonic=True, error=False, num_input=2):
    '''Return validation and training loss data over epochs 
    Source: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch'''
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
    model = get_NN_model(monotonic, error=error,num_input=num_input)

    # list of val loss and train loss data for plotting
    val_loss, train_loss = [], []
    for epoch in range(EPOCHS):
        start_time = time.time()
        print("\nStart of epo ch %d" % (epoch,))
        
        # iterate over batches - note: x_batch_train has shape (BATCH_SIZE * 10 * 2) and y_batch_train has shape (BATCH_SIZE)
        for step, (x_batch_train, y_batch_train) in enumerate(zip(X_train_batched, Y_train_batched)):
            loss_value = train_step(x_batch_train, y_batch_train, model, error=error, num_input=num_input)
            # appending was here
            # log every 10 batches - note: for training we have 60 batches and for validation we have 20 batches
            # print("STEP: ", step, len(X_train_batched), len(Y_train_batched))
            if (step % 100 == 0):
                #print(model.get_layer('dense_1').get_weights()[0])
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))
        # run validation loop at the end of each epoch.
        for (x_batch_valid, y_batch_valid) in zip(X_valid_batched, Y_valid_batched):
            val_loss_value = val_step(x_batch_valid, y_batch_valid, model, error=error,num_input=num_input)
            # appending was here
        train_loss.append(loss_value) # save train loss for plotting
        val_loss.append(val_loss_value) # save val loss for plotting
        print("Time taken: %.2fs" % (time.time() - start_time))
    # >FIXME add defn for test_pred after fixing the speed issue
    #test_pred = get_NN_pred(model, Xo_samp_test, error) if error else get_NN_pred(model, X_test, error)
    return model, train_loss, val_loss


# def differentiate_A(model, r, phi, delta_r):
#     '''Return dA(r, phi)/r ie. return predicted integrated dust at (r - delta_r, phi) subtracted from predicted integrated dust at (r, phi)
#     note: this is a single data point'''
#     pred1 = get_NN_pred(model, [(r, phi)], error=False).numpy() # NN prediction for integrated dust at r, phi
#     pred2 = get_NN_pred(model, [(r - delta_r, phi)], error=False).numpy() # NN prediction for integrated dust at r - delta_r, phi
#     print(pred1, pred2)
#     return pred1 - pred2

# def plot_true_vs_predicted_dust(model, error=False):
#     # differentiate(predictions when given Xo_test)
#     for r, phi in X_test:
#         test_pred_differentiated = differentiate_A(model, r, phi, delta_r)
#     # true dust
#     true_dust = np.exp(logdust(Xo_test))
#     # >FIXME move this to the general_ploting functions: redundant
#     plt.scatter(true_dust, test_pred_differentiated, s=1, color='blue', linestyle='-', linewidth = 0.1, marker = 'D', edgecolor='black') 
#     plt.xlabel('Intrinsic dust density \rho(\r, \theita)')
#     plt.ylabel('Predicted dust density \rho(\r, \theita)')
#     # lims = [min(np.amin(Yo_data), np.amin(pred)), max(np.amax(Yo_data), np.amax(pred))]
#     # plt.xlim(lims)
#     # plt.ylim(lims)
#     # plt.plot(lims, lims)
#     plt.title(title)
         
# # call on plot_true_vs_predicted