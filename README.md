# Building the Next Generation of 3D Dust Maps
### How functionality is organized: .py file content
**general_plotting.py**: Includes functions to plot predictions vs true, plot a 3 panel scatterplot (of true, predictions, and residual values), and plot noisy vs intrinsic data.

**imports.py**: Includes nessesary imports and default plotting options

**loading_data.py:** Loads the data and defines constant variables.

**loss_functions.py:** Includes custom lin and asinh loss functions used in the bin, interpolation, and neural network models.

**fixed_bins.py:** Includes functions to get a bin model, plot the loss as the bin numbers increase from a given min to max bin number, and get the dust density predictions of a bin model.

**interpolation.py:** Includes functions to get the interpolation model, and plot the interpolation loss as the smoothing increases from e^-3 to e^5, and get the dust density predictions of the interpolation model (yet to be added).

**neural_network.py** Includes functions to get NN model, a custom loss function (I have to move this to loss_functions later), plot training and validation loss of the NN as it is being trained, and get the dust density predictions of the NN model (yet to be added).
