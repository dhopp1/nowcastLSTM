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
  py_run_string("from nowcast_lstm.model_selection import variable_selection, hyperparameter_tuning, select_model")
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
#' @param decay double, learning rate decay
#' @param n_hidden integer, number of hidden states in the network
#' @param n_layers integer, number of LSTM layers in the network
#' @param dropout double, dropout rate between the LSTM layers
#' @param criterion torch loss criterion, defaults to MAE. For E.g. MSE, pass "torch.nn.MSELoss()"
#' @param optimizer torch optimizer, defaults to Adam. For a different one, pass e.g. "torch.optim.SGD"
#' @param optimizer_parameters parameters for optimizer, including learning rate. Pass as a named list, e.g. list("lr"=0.01, "weight_decay"=0.001}
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
            decay = 0.98,
            n_hidden = 20,
            n_layers = 2,
            dropout = 0,
            criterion = "''",
            optimizer = "''",
            optimizer_parameters = list("lr"=1e-2),
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
    
    # converting R named list to python Dict
    list_to_dict <- function (my_list) {
      return (paste0("{", paste(paste0("'", names(my_list), "':", my_list), collapse=","), "}"))
    }
    optimizer_parameters_dict <- list_to_dict(optimizer_parameters)
    
    py_run_string(
      str_interp(
        "${python_model_name} = LSTM(data=r.tmp_df, target_variable='${target_variable}', n_timesteps=${n_timesteps}, fill_na_func=${fill_na_func}, fill_ragged_edges_func=${fill_ragged_edges_func}, n_models=${n_models}, train_episodes=${train_episodes}, batch_size=${batch_size}, decay=${decay}, n_hidden=${n_hidden}, n_layers=${n_layers}, dropout=${dropout}, criterion=${criterion}, optimizer=${optimizer}, optimizer_parameters=${optimizer_parameters_dict})"
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
  r_news <- list()
  
  # convert to R dataframe if not already one
  if (typeof(news$news) != "list") {
    r_news[["news"]] <- py_to_r(news$news) 
  } else {
    r_news[["news"]] <- news$news
  }
  r_news[["old_pred"]] <- news$old_pred
  r_news[["new_pred"]] <- news$new_pred
  r_news[["holdout_discrepency"]] <- news$holdout_discrepency
  
  return (r_news)
}


#' @title Variable selection
#' @description Pick best-performing variables for a given set of hyperparameters. All parameters before `n_folds` identical to a base LSTM model.
#' @param n_folds how many folds for rolling fold validation to do
#' @param init_test_size ϵ [0,1]. What proportion of the data to use for testing at the first fold
#' @param pub_lags list of periods back each input variable is set to missing. I.e. publication lag of the variable. Leave empty to pick variables only on complete information, no synthetic vintages.
#' @param lags simulated periods back to test when selecting variables. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc. So [-2, 0, 2] will account for those vintages in model selection. Leave empty to pick variables only on complete information, no synthetic vintages.
#' @param performance_metric performance metric to use for variable selection. Pass "RMSE" for root mean square error, "MAE" for mean absolute error, or "AICc" for correctd Akaike Information Criterion. Alternatively can pass a function that takes arguments of a pandas Series of predictions and actuals and returns a scalar. E.g. custom_function(preds, actuals).
#' @param alpha ϵ [0,1]. 0 implies no penalization for additional regressors, 1 implies most severe penalty for additional regressors. Not used for "AICc" performance metric.
#' @param initial_ordering ϵ ["feature_contribution", "univariate"]. How to get initial order of features to check additively. "feature_contribution" uses the feature contribution of one model, "univariate" calculates univariate models of all features and orders by performance metric. Feature contribution is about twice as fast.
#' @param quiet whether or not to print progress
#' @return A \code{list} containing the following elements:
#'
#' \item{col_names}{list of best-performing column names}
#' \item{performance}{performance metric of these variables (i.e. best performing)}
#'
#' @export

variable_selection <- function (
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
  optimizer_parameters = list("lr"=1e-2),
  n_folds = 1,
  init_test_size = 0.2,
  pub_lags = c(),
  lags = c(),
  performance_metric = "RMSE",
  alpha = 0.0,
  initial_ordering="feature_contribution",
  quiet = FALSE
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
    } else if (x == "RMSE") {
      return ('"RMSE"')
    } else if (x == "MAE") {
      return ('"MAE"')
    } else if (x == "AICc") {
      return ('"AICc"')
    } else if (x == TRUE) {
      return ('True')
    } else if (x == FALSE) {
      return ('False')
    }
  }
  fill_na_func <- fill_switch(fill_na_func)
  fill_ragged_edges_func <- fill_switch(fill_ragged_edges_func)
  performance_metric <- fill_switch(performance_metric)
  quiet <- fill_switch(quiet)
  
  # converting R named list to python Dict
  list_to_dict <- function (my_list) {
    return (paste0("{", paste(paste0("'", names(my_list), "':", my_list), collapse=","), "}"))
  }
  optimizer_parameters_dict <- list_to_dict(optimizer_parameters)
  
  # converting R vector to python list
  vec_to_list <- function (my_vec) {
    if (is_empty(my_vec)) {
      return ("[]")
    } else {
      final_string = "["
      for (i in my_vec) {
        final_string <- paste0(final_string, i, ",")
      }
      final_string <- paste0(final_string, "]")
      return (final_string)
    }
  }
  pub_lags <- vec_to_list(pub_lags)
  lags <- vec_to_list(lags)
  
  py_run_string(
    str_interp(
      "tmp1, tmp2 = variable_selection(data=r.tmp_df, target_variable='${target_variable}', n_timesteps=${n_timesteps}, fill_na_func=${fill_na_func}, fill_ragged_edges_func=${fill_ragged_edges_func}, n_models=${n_models}, train_episodes=${train_episodes}, batch_size=${batch_size}, decay=${decay}, n_hidden=${n_hidden}, n_layers=${n_layers}, dropout=${dropout}, criterion=${criterion}, optimizer=${optimizer}, optimizer_parameters=${optimizer_parameters_dict}, n_folds=${n_folds}, init_test_size=${init_test_size}, pub_lags=${pub_lags}, lags=${lags}, performance_metric=${performance_metric}, alpha=${alpha}, initial_ordering=${initial_ordering}, quiet=${quiet})"
    )
  )
  
  return (
    list(col_names = py$tmp1, performance = py$tmp2)
    )
}


#' @title Hyperparameter tuning
#' @description Pick best-performing hyperparameters for a given dataset. n_timesteps_grid has default grid for predicting quarterly variable with monthly series, may have to change per use case. E.g. [12,24] for a yearly target with monthly indicators. All parameters up to `optimizer_parameters` exactly the same as for any LSTM() model, provide a list with the values to check.
#' @param n_folds how many folds for rolling fold validation to do
#' @param init_test_size ϵ [0,1]. What proportion of the data to use for testing at the first fold
#' @param pub_lags list of periods back each input variable is set to missing. I.e. publication lag of the variable. Leave empty to pick variables only on complete information, no synthetic vintages.
#' @param lags simulated periods back to test when selecting variables. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc. So [-2, 0, 2] will account for those vintages in model selection. Leave empty to pick variables only on complete information, no synthetic vintages.
#' @param performance_metric performance metric to use for variable selection. Pass "RMSE" for root mean square error, "MAE" for mean absolute error, or "AICc" for correctd Akaike Information Criterion. Alternatively can pass a function that takes arguments of a pandas Series of predictions and actuals and returns a scalar. E.g. custom_function(preds, actuals).
#' @return A \code{dataframe} containing the following elements:
#'
#' \item{hyper_params}{liste of hyperparameters, access via df$hyper_params[[1]], etc.}
#' \item{performance}{performance metric of these hyperparameteres}
#'
#' @export

hyperparameter_tuning <- function (
  data,
  target_variable,
  n_models = 1,
  n_timesteps_grid = c(6, 12),
  fill_na_func_grid = c("mean"),
  fill_ragged_edges_func_grid = c("mean"),
  train_episodes_grid = c(50, 100, 200),
  batch_size_grid = c(30, 100, 200),
  decay_grid = c(0.98),
  n_hidden_grid = c(10, 20, 40),
  n_layers_grid = c(1, 2, 4),
  dropout_grid = c(0),
  criterion_grid = c("''"),
  optimizer_grid = c("''"),
  optimizer_parameters_grid = c(list("lr"=1e-2)),
  n_folds = 1,
  init_test_size = 0.2,
  pub_lags = c(),
  lags = c(),
  performance_metric = "RMSE",
  quiet = FALSE
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
    } else if (x == "RMSE") {
      return ('"RMSE"')
    } else if (x == "MAE") {
      return ('"MAE"')
    } else if (x == "AICc") {
      return ('"AICc"')
    } else if (x == TRUE) {
      return ('True')
    } else if (x == FALSE) {
      return ('False')
    }
  }
  performance_metric <- fill_switch(performance_metric)
  quiet <- fill_switch(quiet)
  
  # converting R named list to python Dict
  list_to_dict <- function (my_list) {
    return (paste0("{", paste(paste0("'", names(my_list), "':", my_list), collapse=","), "}"))
  }
  
  # converting R vector to python list
  vec_to_list <- function (my_vec) {
    if (is_empty(my_vec)) {
      return ("[]")
    } else {
      final_string = "["
      for (i in my_vec) {
        final_string <- paste0(final_string, i, ",")
      }
      final_string <- paste0(final_string, "]")
      return (final_string)
    }
  }
  pub_lags <- vec_to_list(pub_lags)
  lags <- vec_to_list(lags)
  n_timesteps_grid <- vec_to_list(n_timesteps_grid)
  for (i in 1:length(fill_na_func_grid)) {
    fill_na_func_grid[i] <- fill_switch(fill_na_func_grid[i])
  }
  fill_na_func_grid <- vec_to_list(fill_na_func_grid)
  for (i in 1:length(fill_ragged_edges_func_grid)) {
    fill_ragged_edges_func_grid[i] <- fill_switch(fill_ragged_edges_func_grid[i])
  }
  fill_ragged_edges_func_grid <- vec_to_list(fill_ragged_edges_func_grid)
  train_episodes_grid <- vec_to_list(train_episodes_grid)
  batch_size_grid <- vec_to_list(batch_size_grid)
  decay_grid <- vec_to_list(decay_grid)
  n_hidden_grid <- vec_to_list(n_hidden_grid)
  n_layers_grid <- vec_to_list(n_layers_grid)
  dropout_grid <- vec_to_list(dropout_grid)
  criterion_grid <- vec_to_list(criterion_grid)
  optimizer_grid <- vec_to_list(optimizer_grid)
  for (i in 1:length(optimizer_parameters_grid)) {
    optimizer_parameters_grid[i] <- list_to_dict(optimizer_parameters_grid[i])
  }
  optimizer_parameters_grid <- vec_to_list(optimizer_parameters_grid)
  
  py_run_string(
    str_interp(
      "tmp1 = hyperparameter_tuning(data=r.tmp_df, target_variable='${target_variable}', n_timesteps_grid=${n_timesteps_grid}, fill_na_func_grid=${fill_na_func_grid}, fill_ragged_edges_func_grid=${fill_ragged_edges_func_grid}, n_models=${n_models}, train_episodes_grid=${train_episodes_grid}, batch_size_grid=${batch_size_grid}, decay_grid=${decay_grid}, n_hidden_grid=${n_hidden_grid}, n_layers_grid=${n_layers_grid}, dropout_grid=${dropout_grid}, criterion_grid=${criterion_grid}, optimizer_grid=${optimizer_grid}, optimizer_parameters_grid=${optimizer_parameters_grid}, n_folds=${n_folds}, init_test_size=${init_test_size}, pub_lags=${pub_lags}, lags=${lags}, performance_metric=${performance_metric}, quiet=${quiet})"
    )
  )
  
  return (py$tmp1)
}


#' @title Variable selection and hyperparameter tuning combined.
#' @description Pick best-performing hyperparameters and variables for a given dataset. Given all permutations of hyperparameters (k), and p variables in the data, this function will run k * p * 2 models. This can take a very long time. To cut down on this time, run it with a highly reduced hyperparameter grid, i.e., a very small k, then record the selected variables, then run the `hyperparameter_tuning` function with these selected varaibles with a much more detailed grid. All parameters up to `optimizer_parameters` exactly the same as for any LSTM() model, provide a list with the values to check.
#' @param n_folds how many folds for rolling fold validation to do
#' @param init_test_size ϵ [0,1]. What proportion of the data to use for testing at the first fold
#' @param pub_lags list of periods back each input variable is set to missing. I.e. publication lag of the variable. Leave empty to pick variables only on complete information, no synthetic vintages.
#' @param lags simulated periods back to test when selecting variables. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc. So [-2, 0, 2] will account for those vintages in model selection. Leave empty to pick variables only on complete information, no synthetic vintages.
#' @param performance_metric performance metric to use for variable selection. Pass "RMSE" for root mean square error, "MAE" for mean absolute error, or "AICc" for correctd Akaike Information Criterion. Alternatively can pass a function that takes arguments of a pandas Series of predictions and actuals and returns a scalar. E.g. custom_function(preds, actuals).
#' @param alpha ϵ [0,1]. 0 implies no penalization for additional regressors, 1 implies most severe penalty for additional regressors. Not used for "AICc" performance metric.
#' @param initial_ordering ϵ ["feature_contribution", "univariate"]. How to get initial order of features to check additively. "feature_contribution" uses the feature contribution of one model, "univariate" calculates univariate models of all features and orders by performance metric. Feature contribution is about twice as fast.
#' @return A \code{dataframe} containing the following elements:
#'
#' \item{variables}{list of variables}
#' \item{hyper_params}{list of hyperparameters, access via df$hyper_params[[1]], etc.}
#' \item{performance}{performance metric of these hyperparameteres}
#'
#' @export

select_model <- function (
  data,
  target_variable,
  n_models = 1,
  n_timesteps_grid = c(6, 12),
  fill_na_func_grid = c("mean"),
  fill_ragged_edges_func_grid = c("mean"),
  train_episodes_grid = c(50, 100, 200),
  batch_size_grid = c(30, 100, 200),
  decay_grid = c(0.98),
  n_hidden_grid = c(10, 20, 40),
  n_layers_grid = c(1, 2, 4),
  dropout_grid = c(0),
  criterion_grid = c("''"),
  optimizer_grid = c("''"),
  optimizer_parameters_grid = c(list("lr"=1e-2)),
  n_folds = 1,
  init_test_size = 0.2,
  pub_lags = c(),
  lags = c(),
  performance_metric = "RMSE",
  alpha=0.0,
  initial_ordering="feature_contribution",
  quiet = FALSE
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
    } else if (x == "RMSE") {
      return ('"RMSE"')
    } else if (x == "MAE") {
      return ('"MAE"')
    } else if (x == "AICc") {
      return ('"AICc"')
    } else if (x == TRUE) {
      return ('True')
    } else if (x == FALSE) {
      return ('False')
    }
  }
  performance_metric <- fill_switch(performance_metric)
  quiet <- fill_switch(quiet)
  
  # converting R named list to python Dict
  list_to_dict <- function (my_list) {
    return (paste0("{", paste(paste0("'", names(my_list), "':", my_list), collapse=","), "}"))
  }
  
  # converting R vector to python list
  vec_to_list <- function (my_vec) {
    if (is_empty(my_vec)) {
      return ("[]")
    } else {
      final_string = "["
      for (i in my_vec) {
        final_string <- paste0(final_string, i, ",")
      }
      final_string <- paste0(final_string, "]")
      return (final_string)
    }
  }
  pub_lags <- vec_to_list(pub_lags)
  lags <- vec_to_list(lags)
  n_timesteps_grid <- vec_to_list(n_timesteps_grid)
  for (i in 1:length(fill_na_func_grid)) {
    fill_na_func_grid[i] <- fill_switch(fill_na_func_grid[i])
  }
  fill_na_func_grid <- vec_to_list(fill_na_func_grid)
  for (i in 1:length(fill_ragged_edges_func_grid)) {
    fill_ragged_edges_func_grid[i] <- fill_switch(fill_ragged_edges_func_grid[i])
  }
  fill_ragged_edges_func_grid <- vec_to_list(fill_ragged_edges_func_grid)
  train_episodes_grid <- vec_to_list(train_episodes_grid)
  batch_size_grid <- vec_to_list(batch_size_grid)
  decay_grid <- vec_to_list(decay_grid)
  n_hidden_grid <- vec_to_list(n_hidden_grid)
  n_layers_grid <- vec_to_list(n_layers_grid)
  dropout_grid <- vec_to_list(dropout_grid)
  criterion_grid <- vec_to_list(criterion_grid)
  optimizer_grid <- vec_to_list(optimizer_grid)
  for (i in 1:length(optimizer_parameters_grid)) {
    optimizer_parameters_grid[i] <- list_to_dict(optimizer_parameters_grid[i])
  }
  optimizer_parameters_grid <- vec_to_list(optimizer_parameters_grid)
  
  py_run_string(
    str_interp(
      "tmp1 = select_model(data=r.tmp_df, target_variable='${target_variable}', n_timesteps_grid=${n_timesteps_grid}, fill_na_func_grid=${fill_na_func_grid}, fill_ragged_edges_func_grid=${fill_ragged_edges_func_grid}, n_models=${n_models}, train_episodes_grid=${train_episodes_grid}, batch_size_grid=${batch_size_grid}, decay_grid=${decay_grid}, n_hidden_grid=${n_hidden_grid}, n_layers_grid=${n_layers_grid}, dropout_grid=${dropout_grid}, criterion_grid=${criterion_grid}, optimizer_grid=${optimizer_grid}, optimizer_parameters_grid=${optimizer_parameters_grid}, n_folds=${n_folds}, init_test_size=${init_test_size}, pub_lags=${pub_lags}, lags=${lags}, performance_metric=${performance_metric}, alpha=${alpha}, initial_ordering=${initial_ordering}, quiet=${quiet})"
    )
  )
  
  return (py$tmp1)
}
