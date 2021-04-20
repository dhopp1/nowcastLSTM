#' @import dplyr
#' @import stringr
#' @import reticulate

#' @title Install required Python libraries
#' @description Installs required Python libraries. If this fails, install `pandas`, `nowcast-lstm`, and `pmdarima` manually from the command line with pip
#' 
#' @export

install_libraries <- function () {
  py_install("pandas", pip=TRUE)
  py_install("numpy", pip=TRUE)
  py_install("nowcast-lstm", pip=TRUE)
  py_install("pmdarima", pip=TRUE)
}


#' @title Initialize Python libraries
#' @description Initializes Python libraries for the session
#' 
#' @export

initialize_session <- function () {
  py_run_string("from nowcast_lstm.LSTM import LSTM")
  py_run_string("import pandas as pd")
  py_run_string("import numpy as np")
}


#' @title Install required Python libraries
#' @description Installs required Python libraries. If this fails, install `pandas`, `nowcast-lstm`, and `pmdarima` manually from the command line with pip
#' @param data matrix of variables, size (n_obs, n_variables). Must include in 1st column a series of type date, called "date", all data already stationary.
#' @param output_dfm list, the output of the \code{dfm()} function.
#' @param months_ahead number of months ahead to forecast.
#' @param lag number of lags for the kalman filter
#' @return dataframe with all missing values filled + predictions.

# convert a dataframe (with column date) to format for predicting and training model
format_dataframe <- function (dataframe, date_col) {
  dataframe_name <- deparse(substitute(dataframe))
  assign(dataframe_name, r_to_py(dataframe), envir=.GlobalEnv)
  py_run_string(str_interp("r.${dataframe_name}.${date_col} = pd.to_datetime(r.${dataframe_name}.${date_col})"))
}


# instantiate and train an LSTM model
LSTM <-
  function (data,
            date_col,
            target_variable,
            n_timesteps,
            fill_na_func = "mean",
            fill_ragged_edges_func = "mean",
            n_models = 1,
            train_episodes = 200,
            batch_size = 30,
            lr = 0.01,
            decay = 0.98,
            n_hidden = 20,
            n_layers = 2,
            dropout = 0,
            criterion = "''",
            optimizer = "''") {
    dataframe_name <- deparse(substitute(data))
    
    format_dataframe(data, date_col)
    # NA and ragged edges filling
    fill_switch <- function (x) {
      if (x == "mean") {
        return ("np.nanmean")
      } else if (x == "median") {
        return ("np.nanmedian")
      } else if (x == "ARMA") {
        return ('"ARMA"')
      }
    }
    fill_na_func <- fill_switch(fill_na_func)
    fill_ragged_edges_func <- fill_switch(fill_ragged_edges_func)
    
    py_run_string(
      str_interp(
        "model = LSTM(data=r.${dataframe_name}, target_variable='${target_variable}', n_timesteps=${n_timesteps}, fill_na_func=${fill_na_func}, fill_ragged_edges_func=${fill_ragged_edges_func}, n_models=${n_models}, train_episodes=${train_episodes}, batch_size=${batch_size}, lr=${lr}, decay=${decay}, n_hidden=${n_hidden}, n_layers=${n_layers}, dropout=${dropout}, criterion=${criterion}, optimizer=${optimizer})"
      )
    )
    py_run_string("model.train(quiet=True)")
    return (py$model)
  }

# predict, make sure turn dates back into R
predict <- function (x) {
  
}

# ragged predict
ragged_preds

# gen news
gen_news

### testing
library(dplyr)
library(stringr)
library(reticulate)

data <- readr::read_csv("/Users/danhopp/dhopp1/UNCTAD/nowcast_data_update/output/2020-03-01_database_tf.csv")

data <- data %>% 
  select(date, x_cn, x_us, x_jp, x_kr, x_world)

initialize_session()

model <- LSTM(data, "date", "x_world", n_timesteps=12)
model$predict(data)