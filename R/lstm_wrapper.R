#' @import dplyr
#' @import stringr
#' @import reticulate

#' @title Install required Python libraries
#' @description Installs required Python libraries. If this fails, install `pandas`, `nowcast-lstm`, and `pmdarima` manually from the command line with pip
#' 
#' @export

install_python_libraries <- function () {
  py_install("pandas", pip=TRUE)
  py_install("numpy", pip=TRUE)
  py_install("nowcast-lstm", pip=TRUE)
  py_install("pmdarima", pip=TRUE)
  py_install("dill", pip=TRUE)
}


#' @title Initialize Python libraries
#' @description Initializes Python libraries for the session
#' 
#' @export

initialize_session <- function () {
  py_run_string("from nowcast_lstm.LSTM import LSTM")
  py_run_string("import pandas as pd")
  py_run_string("import numpy as np")
  py_run_string("import dill")
}


#' @title Install required Python libraries
#' @description Installs required Python libraries. If this fails, install `pandas`, `nowcast-lstm`, and `pmdarima` manually from the command line with pip
#' @param data matrix of variables, size (n_obs, n_variables). Must include in 1st column a series of type date, called "date", all data already stationary.
#' @param output_dfm list, the output of the \code{dfm()} function.
#' @param months_ahead number of months ahead to forecast.
#' @param lag number of lags for the kalman filter
#' @return dataframe with all missing values filled + predictions.

# convert a dataframe (with column date) to format for predicting and training model
format_dataframe <- function (dataframe, df_name="tmp_df") {
  date_col <- colnames(dataframe[sapply(dataframe, class) == "Date"])[1]
  assign(df_name, r_to_py(dataframe), envir=.GlobalEnv)
  py_run_string(str_interp("r.${df_name}.${date_col} = pd.to_datetime(r.${df_name}.${date_col})"))
}


# instantiate and train an LSTM model
LSTM <-
  function (data,
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
            optimizer = "''",
            python_model_name = "model"
            ) {
    format_dataframe(data)
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
        "${python_model_name} = LSTM(data=r.tmp_df, target_variable='${target_variable}', n_timesteps=${n_timesteps}, fill_na_func=${fill_na_func}, fill_ragged_edges_func=${fill_ragged_edges_func}, n_models=${n_models}, train_episodes=${train_episodes}, batch_size=${batch_size}, lr=${lr}, decay=${decay}, n_hidden=${n_hidden}, n_layers=${n_layers}, dropout=${dropout}, criterion=${criterion}, optimizer=${optimizer})"
      )
    )
    py_run_string(str_interp("${python_model_name}.train(quiet=True)"))
    return (eval(parse(text=str_interp("py$${python_model_name}"))))
  }

predict <- function (model, data, only_actuals_obs = FALSE) {
  date_col <- colnames(data[sapply(data, class) == "Date"])[1]
  format_dataframe(data)
  preds <- model$predict(tmp_df, only_actuals_obs)
  preds <- data.frame(preds)
  preds[,date_col] <- as.Date(preds[,date_col])
  return (preds)
}

# save a model
save_lstm <- function (model, path, python_model_name="") {
  if (python_model_name == "") {
    python_model_name <- deparse(substitute(model))  
  }
  py_run_string(str_interp("dill.dump(${python_model_name}, open('${path}', mode='wb'))"))
}

# load a model
load_lstm <- function (path, python_model_name="load_tmp_model") {
 py_run_string(str_interp("${python_model_name} = dill.load(open('${path}', 'rb', -1))"))
  return (eval(parse(text=str_interp("py$${python_model_name}"))))
}

# ragged predict
ragged_preds <- function (model, pub_lags, lag, data, start_date = NULL, end_date = NULL) {
  date_col <- colnames(data[sapply(data, class) == "Date"])[1]
  format_dataframe(data)
  preds <- model$ragged_preds(as.integer(pub_lags), as.integer(lag), tmp_df, start_date, end_date)
  preds <- data.frame(preds)
  preds[,date_col] <- as.Date(preds[,date_col])
  return (preds)
}

# gen news
gen_news <- function (model, target_period, old_data, new_data) {
  format_dataframe(old_data, "news_old")
  format_dataframe(new_data, "news_new")
  news <- model$gen_news(target_period, news_old, news_new)
  return (news)
}

### testing
library(dplyr)
library(stringr)
library(reticulate)

data <- readr::read_csv("/Users/danhopp/dhopp1/UNCTAD/nowcast_data_update/output/2020-03-01_database_tf.csv")
data <- data %>% 
  select(date, x_cn, x_us, x_jp, x_kr, x_world)

initialize_session()

test <- LSTM(data, "x_world", n_timesteps=12, python_model_name = "test")
predict(test, data, TRUE)
ragged_preds(test, list(0,1,2,3), -1, data)
save_lstm(test, "model2.pkl")
x <- load_lstm("model2.pkl")
y <- load_lstm("model2.pkl", "y")

gen_news(test, "2020-03-01", filter(data, date <= "2019-12-01"), data)