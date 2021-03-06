% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{TREECpp}
\alias{TREECpp}
\title{Performs a two-sample Bayesian nonparametric Polya tree test between two independent univariate samples}
\usage{
TREECpp(x1, x0, cval, m, s, DOWN, UP, depth, maxdepth = 10L)
}
\arguments{
\item{x1}{training sample from group 1}

\item{x0}{training sample from group 0}

\item{cval}{smoothing parameter}

\item{m}{mean of centering Gaussian distribution}

\item{s}{sd of centering Gaussian distribution}

\item{DOWN}{Leave this number as 0}

\item{UP}{Leave this number as 1}

\item{depth}{Leave this number as 0}

\item{maxdepth}{Maximum depth to truncate the tree. Default is 10.}
}
\value{
log Bayes factor
}
\description{
Performs a two-sample Bayesian nonparametric Polya tree test between two independent univariate samples
}
