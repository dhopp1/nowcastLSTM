# nowcastLSTM
R wrapper for [nowcast_lstm](https://github.com/dhopp1/nowcast_lstm) Python library. Long short-term memory neural networks for economic nowcasting. More background in [this](https://unctad.org/webflyer/economic-nowcasting-long-short-term-memory-artificial-neural-networks-lstm) UNCTAD research paper.

# Installation and set up
**Installing the library**: Install devtools with `install.packages("devtools")`. Then, from R, run: `devtools::install_github("dhopp1/nowcastLSTM")`. If you get errors about packages being built on different versions of R, try running `Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS="true")`, then run the install command again.
<br><br>
**Installing Python**: If you already have Python installed on your system, simply follow the install instructions from the [nowcast_lstm Python library](https://github.com/dhopp1/nowcast_lstm) and point `initialize_session` to the path where your Python is installed later on.
<br><br>
If you don't have Python installed on your system, run the following commands in R once you've run `devtools::install_github("dhopp1/nowcastLSTM")`:

```R
library(reticulate)
install_miniconda(path = miniconda_path(), update = TRUE, force = FALSE)
py_install(conda=miniconda_path(), "dill numpy pandas pmdarima torch nowcast-lstm", pip=TRUE)
```

**Example**: `nowcastLSTM_example.zip` contains an R Markdown file with a dataset and more detailed example of usage in R.

## Set up
Once all Python libraries are installed, run the `initialize_session` function in R each time you use the library.

```R
library(nowcastLSTM)

# this function should be run at the beginning of every session. Python path can be left empty to use the system default
initialize_session(python_path = "path_to/python")

# if you installed Python via reticulate, use this. You may get a warning about requesting one path and getting another, but it should work regardless.
initialize_session(python_path = miniconda_path())

# use this to set Python location permanently
Sys.setenv(RETICULATE_PYTHON = "path_to/python")
```

## Background
[LSTM neural networks](https://en.wikipedia.org/wiki/Long_short-term_memory) have been used for nowcasting [before](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf), combining the strengths of artificial neural networks with a temporal aspect. However their use in nowcasting economic indicators remains limited, no doubt in part due to the difficulty of obtaining results in existing deep learning frameworks. This library seeks to streamline the process of obtaining results in the hopes of expanding the domains to which LSTM can be applied.

While neural networks are flexible and this framework may be able to get sensible results on levels, the model architecture was developed to nowcast growth rates of economic indicators. As such training inputs should ideally be stationary and seasonally adjusted.

Further explanation of the background problem can be found in [this UNCTAD research paper](https://unctad.org/system/files/official-document/ser-rp-2018d9_en.pdf). Further explanation and results in [this](https://unctad.org/webflyer/economic-nowcasting-long-short-term-memory-artificial-neural-networks-lstm) UNCTAD research paper.


## Quick usage
Given `data` = a dataframe with a date column + monthly data + a quarterly target series to run the model on, usage is as follows:

```R
library(nowcastDFM)
initialize_session()

# this command will instantiate and train an LSTM network
# due to quirks with using Python from R, the python_model_name argument should be set to the same name used for the R object it is assigned to.
model <- LSTM(data, "target_col_name", n_timesteps=12, python_model_name = "model") # default parameters with 12 timestep history

predict(model, data) # predictions on the training set

# predicting on a testset, which is the same dataframe as the training data + newer data
# this will give predictions for all dates, but only predictions after the training data ends should be considered for testing
predict(model, test_data)

# to gauge performance on artificial data vintages
ragged_preds(model, pub_lags, lag, test_data)

# save a trained model
# python_model_name should be the same value used when the model was initially trained
save_lstm(model, "trained_model.pkl", python_model_name = "model")

# load a previously trained model
# due to quirks with using Python from R, the python_model_name argument should be set to the same name used for the R object it is assigned to.
trained_model <- load_lstm("trained_model.pkl", python_model_name = "trained_model")
```

## LSTM parameters
- `data`: `dataframe` of the data to train the model on. Should contain a target column. Any non-numeric columns will be dropped. It should be in the most frequent period of the data. E.g. if I have three monthly variables, two quarterly variables, and a quarterly series, the rows of the dataframe should be months, with the quarterly values appearing every three months (whether Q1 = Jan 1 or Mar 1 depends on the series, but generally the quarterly value should come at the end of the quarter, i.e. Mar 1), with NAs or 0s in between. The same logic applies for yearly variables.
- `target_variable`: a `string`, the name of the target column in the dataframe.
- `n_timesteps`: an `int`, corresponding to the "memory" of the network, i.e. the target value depends on the x past values of the independent variables. For example, if the data is monthly, `n_timesteps=12` means that the estimated target value is based on the previous years' worth of data, 24 is the last two years', etc. This is a hyper parameter that can be evaluated.
- `fill_na_func`: a function used to replace missing values. Options are `c("mean", "median", "ARMA")`.
- `fill_ragged_edges_func`: a function used to replace missing values at the end of series. Leave blank to use the same function as `fill_na_func`, pass `"ARMA"` to use ARMA estimation using `pmdarima.arima.auto_arima`. Options are `c("mean", "median", "ARMA")`.
- `n_models`: `int` of the number of networks to train and predict on. Because neural networks are inherently stochastic, it can be useful to train multiple networks with the same hyper parameters and take the average of their outputs as the model's prediction, to smooth output.
- `train_episodes`: `int` of the number of training episodes/epochs. A short discussion of the topic can be found [here](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/).
- `batch_size`: `int` of the number of observations per batch. Discussed [here](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
- `decay`: `float` of the rate of decay of the learning rate. Also discussed [here](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/). Set to `0` for no decay.
- `n_hidden`: `int` of the number of hidden states in the LSTM network. Discussed [here](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/).
- `n_layers`: `int` of the number of LSTM layers to include in the network. Also discussed [here](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/).
- `dropout`: `float` of the proportion of layers to drop in between LSTM layers. Discussed [here](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/).
- `criterion`: `PyTorch loss function`. Discussed [here](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/), list of available options in PyTorch [here](https://pytorch.org/docs/stable/nn.html#loss-functions). Pass as a string, e.g. one of `c("torch.nn.L1Loss()", "torch.nn.MSELoss()")`, etc.
- `optimizer`: `PyTorch optimizer`. Discussed [here](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6), list of available options in PyTorch [here](https://pytorch.org/docs/stable/optim.html). Pass as a string, e.g. `"torch.optim.Adam"`.
- `optimizer_parameters`: `named list`. Parameters for a particular optimizer, including learning rate. Information [here](https://pytorch.org/docs/stable/optim.html). For instance, to change learning rate (default 1e-2), pass `list("lr"=1e-2)`, or weight_decay for L2 regularization, pass `list("lr"=1e-2, "weight_decay"=0.001)`. Learning rate discussed [here](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/).

## LSTM outputs
Assuming a model has been instantiated and trained with `model = LSTM(...)`, the following functions are available, run `help(function)` on any of them to find out more about them and their parameters. Other information, like training loss, is available in the trained `model` object, accessed via `$`, e.g. `model$train_loss`:

- `predict`: to generate predictions on new data
- `save_lstm`: to save a trained model to disk
- `load_lstm`: to load a saved model from disk
- `ragged_preds(model, pub_lags, lag, new_data, start_date, end_date)`: adds artificial missing data then returns a dataframe with date, actuals, and predictions. This is especially useful as a testing mechanism, to generate datasets to see how a trained model would have performed at different synthetic vintages or periods of time in the past. `pub_lags` should be a list of ints (in the same order as the columns of the original data) of length n\_features (i.e. excluding the target variable) dictating the normal publication lag of each of the variables. `lag` is an int of how many periods back we want to simulate being, interpretable as last period relative to target period. E.g. if we are nowcasting June, `lag = -1` will simulate being in May, where May data is published for variables with a publication lag of 0. It will fill with missings values that wouldn't have been available yet according to the publication lag of the variable + the lag parameter. It will fill missings with the same method specified in the `fill_ragged_edges_func` parameter in model instantiation.
- `gen_news(model, target_period, old_data, new_data)`: Generates news between one data release to another, adding an element of causal inference to the network. Works by holding out new data column by column, recording differences between this prediction and the prediction on full data, and registering this difference as the new data's contribution to the prediction. Contributions are then scaled to equal the actual observed difference in prediction in the aggregate between the old dataset and the new dataset.