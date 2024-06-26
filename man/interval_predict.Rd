% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/lstm_wrapper.R
\name{interval_predict}
\alias{interval_predict}
\title{Get predictions plus uncertainty intervals on a new dataset}
\usage{
interval_predict(
  model,
  data,
  interval = 0.95,
  only_actuals_obs = FALSE,
  start_date = NULL,
  end_date = NULL,
  data_availability_weight_scheme = "fc"
)
}
\arguments{
\item{model}{a trained LSTM model gotten from calling LSTM()}

\item{data}{dataframe with the same columns the model was trained on}

\item{interval}{number between 0 and 1, uncertainty interval. A closer number to one gives a higher uncertainty interval. E.g., 0.95 (95%) will give larger bands than 0.5 (50%)}

\item{only_actuals_obs}{whether or not to produce predictions for periods without an actual. E.g. FALSE will return predictions for months in between quarters, even if the target variable is quarterly.}

\item{start_date}{string in "YYYY-MM-DD" format, start date of generating predictions. To save calculation time, i.e. just calculating after testing date instead of all dates}

\item{end_date}{string in "YYYY-MM-DD" format, end date of generating predictions}

\item{data_availability_weight_scheme}{string, weighting scheme for data avilability. "fc" for weighting variables by feature contribution, "equal" for weighting each equally.}
}
\value{
dataframe with periods, actuals if available, point predictions, lower and upper uncertainty intervals
}
\description{
Get predictions plus uncertainty intervals on a new dataset
}
