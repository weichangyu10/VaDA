% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{VNPDA}
\alias{VNPDA}
\title{Fits the VNPDA model with a distinct smoothing parameter for each variable}
\usage{
VNPDA(xtrain, ytrain, Xtest, cval, ay, by, ag, bg, maxdepth = 10L)
}
\arguments{
\item{xtrain}{Variables of training sample (matrix)}

\item{ytrain}{Group labels of training sample}

\item{Xtest}{Variables of testing sample for classification (matrix)}

\item{cval}{Vector of smoothing parameter values}

\item{ay}{Hyparameter 1 of Beta prior for rho_y}

\item{by}{Hyparameter 2 of Beta prior for rho_y}

\item{ag}{Hyparameter 1 of Beta prior for rho_gamma}

\item{bg}{Hyparameter 2 of Beta prior for rho_gamma}

\item{maxdepth}{Maximum depth to truncate the tree. Default is 10.}
}
\value{
List of variable selection and classification results
}
\description{
Fits the VNPDA model with a distinct smoothing parameter for each variable
}
