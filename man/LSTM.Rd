% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/lstm_wrapper.R
\name{LSTM}
\alias{LSTM}
\title{Instantiate and train an LSTM model}
\usage{
LSTM(
  data,
  target_variable,
  n_timesteps,
  fill_na_func = "mean",
  fill_ragged_edges_func = "mean",
  n_models = 1,
  train_episodes = 200,
  batch_size = 30,
  decay = 0.98,
  n_hidden = 20,
  n_layers = 2,
  dropout = 0,
  criterion = "''",
  optimizer = "''",
  optimizer_parameters = list(lr = 0.01),
  seeds = c(),
  python_model_name = "model"
)
}
\arguments{
\item{data}{n x m+1 dataframe with a column of type Date}

\item{target_variable}{string with the column name of the target variable}

\item{n_timesteps}{how many historical periods to consider when training the model. For example if the original data is monthly, n_steps=12 would consider data for the last year.}

\item{fill_na_func}{function to replace within-series NAs. Options are c("mean", "median", "ARMA").}

\item{fill_ragged_edges_func}{function to replace NAs in ragged edges (data missing at end of series). Options are c("mean", "median", "ARMA"). Note ARMA filling will be significantly slower as models have to be estimated for each variable to fill ragged edges.}

\item{n_models}{integer, number of models to train and take the average of for more robust estimates}

\item{train_episodes}{integer, number of epochs/episodes to train the model}

\item{batch_size}{integer, number of observations per training batch}

\item{decay}{double, learning rate decay}

\item{n_hidden}{integer, number of hidden states in the network}

\item{n_layers}{integer, number of LSTM layers in the network}

\item{dropout}{double, dropout rate between the LSTM layers}

\item{criterion}{torch loss criterion, defaults to MAE. For E.g. MSE, pass "torch.nn.MSELoss()"}

\item{optimizer}{torch optimizer, defaults to Adam. For a different one, pass e.g. "torch.optim.SGD"}

\item{optimizer_parameters}{parameters for optimizer, including learning rate. Pass as a named list, e.g. list("lr"=0.01, "weight_decay"=0.001)}

\item{seeds}{c(int), list of integers, what to seed the initial weights for reproducibility. Must be list of same length as n_models parameter}

\item{python_model_name}{what the model will be called in the python session. Relevant if more than one model is being trained for simultaneous use. For clarity, should be the same as the name of the R object the model is being saved to.}
}
\value{
trained LSTM model
}
\description{
Instantiate and train an LSTM model
}
