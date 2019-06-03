#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

#include <cmath>

using namespace Rcpp;
using namespace arma;

//' @title Calculating differences between two sets
//'
//' @param x        vector of elements in first set
//' @param y        vector of elements in second set
//'
//' @return         Vector of set differences
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
//' @examples
//' x <- c(1,3,4,5,8); y <- c(1, 2, 4, 5, 8)
//' arma_setdiff(x,y)
// [[Rcpp::export]]
arma::vec arma_setdiff(arma::vec x, arma::vec y){

    x = arma::unique(x);
    y = arma::unique(y);

    for (size_t j = 0; j < y.n_elem; j++) {
        arma::uvec q1 = arma::find(x == y[j]);
        if (!q1.empty()) {
            x.shed_row(q1(0));
        }
    }

    arma::vec x2 = x;
    return x2;
}

//' @title Applies the expit (inverse logit) function.
//'
//' @param v        vector of input
//'
//' @return         Vector of output
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
//' @examples
//' x <- rt(100,1)
//' expit(x)
// [[Rcpp::export]]
arma::vec expit(const arma::vec& v){

	arma::vec ans(v.n_elem);
	for(int i = 0; i < v.n_elem; i++){

		ans(i) = 1/(1+exp(-v(i)));

	}
	return ans;

}

//' @title Applies the epsilon function in Yu et. al. (2018).
//'
//' @param v        vector of input
//'
//' @return         Vector of output
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
//' @examples
//' x <- rt(1,10)
//' epsilonC(x)
// [[Rcpp::export]]
double epsilonC(const double& v){

	double ans;
  ans = lgamma(v) + v - v*log(v) - 0.5*log(2*M_PI);

	return ans;

}

//' @title Fit the VLDA model in Yu et. al. (2018).
//'
//' @param vy        Class labels of training sample
//' @param mX        Variables of training sample
//' @param mXtest        Variables of testing sample (for prediction)
//' @param r        tuning parameter 1 for penalty  on log likelihood
//' @param kappa        tuning parameter 2 for penalty  on log likelihood
//'
//' @return         List containing variable selection probabilities, classification probabilitiew, likelihood ratio statistics and classification results
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
//' @examples
//' data(LIHCtoy)
//' X <- as.matrix(X)
//' resVLDA <- VLDA(vy = y[1:90], mX = X[1:90,], mXtest = X[91:109,], r = 0.98, kappa = (1/7))
//' resVLDA
// [[Rcpp::export]]
List VLDA(const arma::vec& vy, const arma::mat& mX, const arma::mat& mXtest, const double& r, const double& kappa){

	const uvec inds1 = find(vy == 1);
	const uvec inds0 = find(vy == 0);
	const int n1 = inds1.n_elem;
	const int n0 = inds0.n_elem;
	const int P = mX.n_cols;
	const int N = mX.n_rows;

	const arma::mat mX1 =  mX.rows(inds1);
	const arma::mat mX0 =  mX.rows(inds0);

	const arma::vec vmuhat = mean(mX, 0).t();
	const arma::vec vmuhat1 = mean(mX1, 0).t();
	const arma::vec vmuhat0 = mean(mX0, 0).t();

	const arma::mat Shat = (N - 1)*var(mX, 0, 0).t();
	const arma::mat Shat1 = (n1 - 1)*var(mX1, 0, 0).t();
	const arma::mat Shat0 = (n0 - 1)*var(mX0, 0, 0).t();
	const int nTest = mXtest.n_rows;

	arma::vec xivec(nTest);
	xivec.fill(0.5);
	arma::vec yhat = zeros<vec>(nTest);
	arma::vec vw = zeros<vec>(P);
	arma::vec vwold = zeros<vec>(P);
	arma::vec vDelta = zeros<vec>(P);
	arma::vec vlambda = zeros<vec>(P);
	arma::vec xstar = zeros<vec>(P);

	const double bgamma = pow(P,2) / sqrt(N) * exp(kappa* N / pow((log(N)),r));

	for(int it = 0; it < 20; it++){

	  vlambda = -2*(N+1)*log(Shat1+Shat0)+2*(N+1)*log(Shat);
	  vDelta = log(accu(vwold) + 1) - log(P -accu(vwold) - 1 + bgamma) - 0.5*log(N+1) + 0.5*vlambda;
	  vw = expit(vDelta);
	  vwold = vw;

	}


	for(int i = 0; i < nTest; i++){

		xstar = mXtest.row(i).t();
    		xivec(i) = log(n1) - log(n0) + 0.5*(N+1)*accu(vw % ((pow((xstar - vmuhat0),2) - pow((xstar - vmuhat1),2)) / (Shat1+Shat0)));
		// cout << log(n1/n0) + 0.5*(N+1)*accu(vw % ((pow((xstar - vmuhat0),2) - pow((xstar - vmuhat1),2)) / (Shat1+Shat0))) << " \n";
    		xivec(i) = 1/(1+ exp(-1.0* xivec(i)));
    		yhat(i) = round(xivec(i));

  	}


  	return Rcpp::List::create(Rcpp::Named("lambdaLRT") = vlambda, Rcpp::Named("omega") = vw, Rcpp::Named("y.hat") = yhat, Rcpp::Named("xi") = xivec);

}

//' @title Fit the VQDA model in Yu et. al. (2018).
//'
//' @param vy        Class labels of training sample
//' @param mX        Variables of training sample
//' @param mXtest        Variables of testing sample (for prediction)
//' @param r        tuning parameter 1 for penalty  on log likelihood
//' @param kappa        tuning parameter 2 for penalty  on log likelihood
//'
//' @return         List containing variable selection probabilities, classification probabilitiew, likelihood ratio statistics and classification results
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
//' @examples
//' data(LIHCtoy)
//' X <- as.matrix(X)
//' resVQDA <- VQDA(vy = y[1:90], mX = X[1:90,], mXtest = X[91:109,], r = 0.98, kappa = (1/7))
//' resVQDA
// [[Rcpp::export]]
List VQDA(const arma::vec& vy, const arma::mat& mX, const arma::mat& mXtest, const double& r, const double& kappa){

	  const uvec inds1 = find(vy == 1);
	  const uvec inds0 = find(vy == 0);
	  const int n1 = inds1.n_elem;
	  const int n0 = inds0.n_elem;
	  const int P = mX.n_cols;
	  const int N = mX.n_rows;

	  const arma::mat mX1 =  mX.rows(inds1);
	  const arma::mat mX0 =  mX.rows(inds0);

	  const arma::vec vmuhat = mean(mX, 0).t();
	  const arma::vec vmuhat1 = mean(mX1, 0).t();
	  const arma::vec vmuhat0 = mean(mX0, 0).t();

	  const arma::mat Shat = (N - 1)*var(mX, 0, 0).t();
	  const arma::mat Shat1 = (n1 - 1)*var(mX1, 0, 0).t();
	  const arma::mat Shat0 = (n0 - 1)*var(mX0, 0, 0).t();
	  const arma::mat Sigmahat = Shat/N;
	  const arma::mat Sigmahat0 = Shat0/n0;
	  const arma::mat Sigmahat1 = Shat1/n1;
	  const int nTest = mXtest.n_rows;
	  double Q, Q1, Q0;

	  const double bgamma = pow(P,2) / sqrt(N) * exp(kappa* N / pow((log(N)),r));

	  arma::vec xivec(nTest);
	  xivec.fill(0.5);
	  arma::vec yhat = zeros<vec>(nTest);
	  arma::vec vw = zeros<vec>(P);
	  arma::vec vwold = zeros<vec>(P);
	  arma::vec vDelta = zeros<vec>(P);
	  arma::vec vlambda = zeros<vec>(P);
	  arma::vec xstar = zeros<vec>(P);


	  for(int it = 0; it < 20; it++){

	    //vlambda = -2*(N+1)*log(Shat1+Shat0)+2*(N+1)*log(Shat);
	    vlambda = N*log(Sigmahat) - n0*log(Sigmahat0) -  n1*log(Sigmahat1);
	    //vDelta = log(accu(vw) + 1) - log(P -accu(vw) - 1 + pow(P,1.5)) - 0.5*log(N+1) + 0.5*vlambda;
	    //vDelta = log(accu(vw) + 1) - log(P -accu(vw) - 1 +pow(P,1.5)) - 3*log(N+1) - 2*epsilonC((N+1)/2) + 2*epsilonC((n0/2)) + 2*epsilonC((n1/2)) + log((n0*n1/2));
	    Q = 0.5*(N+1);
	    Q1 = 0.5*n1;
	    Q0 = 0.5*n0;
	    vDelta = log(accu(vwold) + 1) - log(P -accu(vwold) - 1 +bgamma) - 3*log(N+1) - 2*epsilonC(Q) + 2*epsilonC(Q0) + 2*epsilonC(Q1) + log(n0*n1*0.5) + vlambda;
	    //vDelta = log(sum(vw) + 1) - log(P -sum(vw) - 1 +P^(3/2)) - 3*log(n+1) + vlambda - 2*epsilon((n+1)/2) + 2*epsilon(n0/2) + 2*epsilon(n1/2) + log(n0*n1/2)
	    vw = expit(vDelta);
	    vwold = vw;
	  }

	  for(int i = 0; i < nTest; i++){

	    xstar = mXtest.row(i).t();

	    xivec(i) = log(n1) - log(n0) + accu(vw%(lgamma(0.5*(n1+1)) - lgamma(0.5*(n0+1)) + lgamma(0.5*n0) - lgamma(0.5*n1) + 0.5*(log(Sigmahat0) - log(Sigmahat1) +pow((xstar - vmuhat0),2) /Sigmahat0 -  pow((xstar - vmuhat1),2) /Sigmahat1   ))) ;
	    xivec(i) = 1 /(1 + exp(-xivec(i)));
	    yhat(i) = round(xivec(i));

	  }
	  //return vw;
	  return Rcpp::List::create(Rcpp::Named("lambdaLRT") = vlambda, Rcpp::Named("omega") = vw, Rcpp::Named("y.hat") = yhat, Rcpp::Named("xi") = xivec);

}

