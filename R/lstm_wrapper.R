#' @import dplyr
#' @import stringr
#' @import reticulate


#' @title Initialize Python libraries
#' @param python_path string, location of Python. If left empty, uses default.
#' @description Initializes Python libraries for the session
#'
#' @export

initialize_session <- function (python_path = "") {
  if (python_path != "") {
    use_python(python_path)
  }
  py_run_string("from nowcast_lstm.LSTM import LSTM")
  py_run_string("import pandas as pd")
  py_run_string("import numpy as np")
  py_run_string("import dill")
  py_run_string("import torch")
}

# gets dates in proper format for python + create temporary python df
format_dataframe <- function (dataframe, df_name = "tmp_df") {
  date_col <-
    colnames(dataframe[sapply(dataframe, class) == "Date"])[1]
  assign(df_name, r_to_py(dataframe), envir = .GlobalEnv)
  py_run_string(
    str_interp(
      "r.${df_name}.${date_col} = pd.to_datetime(r.${df_name}.${date_col})"
    )
  )
}


#' @title Instantiate and train an LSTM model
#' @description Instantiate and train an LSTM model
#' @param data n x m+1 dataframe with a column of type Date
#' @param target_variable string with the column name of the target variable
#' @param n_timesteps how many historical periods to consider when training the model. For example if the original data is monthly, n_steps=12 would consider data for the last year.
#' @param fill_na_func function to replace within-series NAs. Options are c("mean", "median", "ARMA").
#' @param fill_ragged_edges_func function to replace NAs in ragged edges (data missing at end of series). Options are c("mean", "median", "ARMA"). Note ARMA filling will be significantly slower as models have to be estimated for each variable to fill ragged edges.
#' @param n_models integer, number of models to train and take the average of for more robust estimates
#' @param train_episodes integer, number of epochs/episodes to train the model
#' @param batch_size integer, number of observations per training batch
#' @param lr double, learning rate
#' @param decay double, learning rate decay
#' @param n_hidden integer, number of hidden states in the network
#' @param n_layers integer, number of LSTM layers in the network
#' @param dropout double, dropout rate between the LSTM layers
#' @param criterion torch loss criterion, defaults to MAE. For E.g. MSE, pass "torch.nn.MSELoss()"
#' @param optimizer torch optimizer, defaults to Adam
#' @param python_model_name what the model will be called in the python session. Relevant if more than one model is being trained for simultaneous use. For clarity, should be the same as the name of the R object the model is being saved to.
#' @return trained LSTM model
#'
#' @export

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
            python_model_name = "model") {
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
    return (eval(parse(text = str_interp(
      "py$${python_model_name}"
    ))))
  }


#' @title Get predictions from a trained LSTM model
#' @description Get predictions from a trained LSTM model
#' @param model a trained LSTM model gotten from calling LSTM()
#' @param data dataframe with the same columns the model was trained on
#' @param only_actuals_obs whether or not to produce predictions for periods without an actual. E.g. FALSE will return predictions for months in between quarters, even if the target variable is quarterly.
#' @return dataframe with periods, actuals if available, and predictions
#'
#' @export

predict <- function (model, data, only_actuals_obs = FALSE) {
  date_col <- colnames(data[sapply(data, class) == "Date"])[1]
  format_dataframe(data)
  preds <- model$predict(tmp_df, only_actuals_obs)
  preds <- data.frame(preds)
  preds[, date_col] <- as.Date(preds[, date_col])
  return (preds)
}


#' @title Save a trained LSTM model to disk
#' @description Save a trained LSTM model to disk
#' @param model a trained LSTM model gotten from calling LSTM()
#' @param path the file name and path to save the model to. Should end in ".pkl"
#' @param python_model_name what the model is called in the python session. Relevant if more than one model is in use. Defaults to same name used for the R object. For clarity, should be the same as the name of the R object the model is being saved to when the `load_lstm` and `LSTM` functions are initially used.
#' @return trained LSTM model
#'
#' @export

save_lstm <- function (model, path, python_model_name = "") {
  if (python_model_name == "") {
    python_model_name <- deparse(substitute(model))
  }
  py_run_string(str_interp(
    "dill.dump(${python_model_name}, open('${path}', mode='wb'))"
  ))
}


#' @title Load a trained LSTM model from disk
#' @description Load a trained LSTM model from disk
#' @param path the file name and path to read the model from. Should end in ".pkl"
#' @param python_model_name what the model will be called in the python session. Relevant if more than one model is being loaded. For clarity, should be the same as the name of the R object the model is being saved to.
#' @return trained LSTM model
#'
#' @export

load_lstm <- function (path, python_model_name = "load_tmp_model") {
  py_run_string(str_interp(
    "${python_model_name} = dill.load(open('${path}', 'rb', -1))"
  ))
  return (eval(parse(text = str_interp(
    "py$${python_model_name}"
  ))))
}


#' @title Get predictions on artificial vintages
#' @description Get predictions on artificial vintages
#' @param model a trained LSTM model gotten from calling LSTM()
#' @param pub_lags list of integers, list of periods back each input variable is set to missing. I.e. publication lag of the variable.
#' @param lag integer, simulated periods back. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc.
#' @param data dataframe to generate the ragged datasets and predictions on
#' @param start_date string in "YYYY-MM-DD" format, start date of generating ragged preds. To save calculation time, i.e. just calculating after testing date instead of all dates
#' @param end_date string in "YYYY-MM-DD" format, end date of generating ragged preds
#' @return dataframe of actuals, predictions, and dates
#'
#' @export

ragged_preds <-
  function (model,
            pub_lags,
            lag,
            data,
            start_date = NULL,
            end_date = NULL) {
    date_col <- colnames(data[sapply(data, class) == "Date"])[1]
    format_dataframe(data)
    preds <-
      model$ragged_preds(as.integer(pub_lags),
                         as.integer(lag),
                         tmp_df,
                         start_date,
                         end_date)
    preds <- data.frame(preds)
    preds[, date_col] <- as.Date(preds[, date_col])
    return (preds)
  }


#' @title Generate the news between two data releases
#' @description Generate the news between two data releases using the method of holding out new data feature by feature and recording the differences in model output
#' @param model a trained LSTM model gotten from calling LSTM()
#' @param target_period string in "YYYY-MM-DD", target prediction date
#' @param old_data dataframe of previous dataset
#' @param new_data dataframe of new dataset
#' @return A \code{list} containing the following elements:
#'
#' \item{news}{dataframe of news contribution of each column with updated data. scaled_news is news scaled to sum to actual prediction delta.}
#' \item{old_pred}{prediction on the previous dataset}
#' \item{new_pred}{prediction on the new dataset}
#' \item{holdout_discrepency}{% difference between the sum of news via the holdout method and the actual prediction delta}
#'
#' @export

gen_news <- function (model, target_period, old_data, new_data) {
  format_dataframe(old_data, "news_old")
  format_dataframe(new_data, "news_new")
  news <- model$gen_news(target_period, news_old, news_new)
  return (news)
}