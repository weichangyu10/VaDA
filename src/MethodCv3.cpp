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

	    vlambda = N*log(Sigmahat) - n0*log(Sigmahat0) -  n1*log(Sigmahat1);
	    Q = 0.5*(N+1);
	    Q1 = 0.5*n1;
	    Q0 = 0.5*n0;
	    vDelta = log(accu(vwold) + 1) - log(P -accu(vwold) - 1 +bgamma) - 3*log(N+1) - 2*epsilonC(Q) + 2*epsilonC(Q0) + 2*epsilonC(Q1) + log(n0*n1*0.5) + vlambda;
	    vw = expit(vDelta);
	    vwold = vw;
	  }

	  for(int i = 0; i < nTest; i++){

	    xstar = mXtest.row(i).t();

	    xivec(i) = log(n1) - log(n0) + accu(vw%(lgamma(0.5*(n1+1)) - lgamma(0.5*(n0+1)) + lgamma(0.5*n0) - lgamma(0.5*n1) + 0.5*(log(Sigmahat0) - log(Sigmahat1) +pow((xstar - vmuhat0),2) /Sigmahat0 -  pow((xstar - vmuhat1),2) /Sigmahat1   ))) ;
	    xivec(i) = 1 /(1 + exp(-xivec(i)));
	    yhat(i) = round(xivec(i));

	  }
	  return Rcpp::List::create(Rcpp::Named("lambdaLRT") = vlambda, Rcpp::Named("omega") = vw, Rcpp::Named("y.hat") = yhat, Rcpp::Named("xi") = xivec);

}

//' @title Calculate classification probabilities for binary Polya tree classification problem with a single feature
//'
//' @param x1       training sample from group 1
//' @param x0       training sample from group 0
//' @param xnew     new data point
//' @param cval     smoothing parameter
//' @param m        mean of centering Gaussian distribution
//' @param s        sd of centering Gaussian distribution
//' @param DOWN     Leave this number as 0
//' @param UP       Leave this number as 1
//' @param depth       Leave this number as 0
//' @param maxdepth    Maximum depth to truncate the tree. Default is 10.
//'
//' @return         Classification probability
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
//' @examples
//' samp1 <- rnorm(50,4,1)
//' samp0 <- rt(40,3)
//' xtest <- rt(1,3)
//' DividePredCpp(x1 = samp1, x0 = samp0, xnew = xtest, cval = 1, m = mean(c(samp1,samp0)), s = sd(c(samp1,samp0)))
// [[Rcpp::export]]
double DividePredCpp(const arma::vec& x1, const arma::vec& x0, const double& xnew, const double& cval, const double& m, const double& s, const double& DOWN = 0, const double& UP = 1, const int& depth = 0, const int& maxdepth = 10) {

  const int n1 = x1.n_elem;
  const int n0 = x0.n_elem;
  if( (n1+n0 == 0) || (depth > maxdepth) ){
    return 0;
  }
  else
  {
    int init_depth = 0;
    if(depth==0) init_depth = 1;
    double alpha = cval*pow(depth, 2) + init_depth;
    const double MIDDLE = (DOWN+UP)/2;
    const double b = R::qnorm(MIDDLE, m, s, TRUE, FALSE);
    uvec x1leftinds = find(x1 <= b);
    uvec x0leftinds = find(x0 <= b);
    const arma::vec x1left = x1.elem( find(x1 <= b) );
    const arma::vec x0left = x0.elem( find(x0 <= b) );
    const arma::vec x1right = arma_setdiff(x1, x1left);
    const arma::vec x0right = arma_setdiff(x0, x0left);
    bool left = xnew <= b;
    int n1left = x1left.n_elem;
    int n0left = x0left.n_elem;
    int n1right = n1 - n1left;
    int n0right = n0 - n0left;
    arma::vec nextx1;
    arma::vec nextx0;
    double ans = left*log(alpha + n1left) - left*log(alpha + n0left) + (1-left)*log(alpha + n1right) - (1-left)*log(alpha + n0right) - log(2*alpha + n1) + log(2*alpha + n0);
    if(left)
    {
      nextx1 = x1left;
      nextx0 = x0left;
    }
    else
    {
      nextx1 = x1right;
      nextx0 = x0right;
    }
    return (ans + DividePredCpp(nextx1,nextx0,xnew,cval,m,s,left*DOWN+(1-left)*MIDDLE,left*MIDDLE+(1-left)*UP,depth+1,maxdepth) );
  }
}

//' @title Calculate classification probabilities for binary Polya tree classification problem with multiple features
//'
//' @param xtrain1  training sample from group 1 (matrix)
//' @param xtrain0  training sample from group 0 (matrix)
//' @param xtest    a single new data point (vector)
//' @param cval     smoothing parameters
//' @param omega    vector of variable selection weights
//' @param ay        hyperparameter 1 of beta prior
//' @param by        hyperparameter 2 of beta prior
//' @param maxdepth    Maximum depth to truncate the tree. Default is 10.
//'
//' @return         Classification probability
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
// [[Rcpp::export]]
double PredCpp(arma::mat xtrain1, arma::mat xtrain0, arma::vec xtest, const arma::vec& cval, arma::vec omega, double ay, double by, int maxdepth = 10) {

  int P = xtest.n_elem;
  int N1 = xtrain1.n_rows;
  int N0 = xtrain0.n_rows;
  arma::vec logContribution = zeros<vec>(P);
  arma::mat xtestMAT = zeros<mat>(1,P);
  xtestMAT.row(0) = xtest.t();
  arma::mat tempMat = join_cols(join_cols(xtrain1, xtrain0), xtestMAT);
  const vec BigMeans = mean( tempMat, 0).t();
  const vec BigSDs = stddev( tempMat , 0).t();
  arma::vec results = zeros<vec>(P);
  double anss = 0;
  for(int j =0; j < P; j++){
    arma::vec x1 = xtrain1.col(j);
    arma::vec x0 = xtrain0.col(j);
    anss = DividePredCpp(x1, x0, xtestMAT(0,j), cval(j), BigMeans(j), BigSDs(j), 0, 1, 0, maxdepth);
    results(j) = anss * omega(j);
  }
  double dataContri = arma::accu(results);
  double priorContri = log(ay + N1) - log(by + N0);
  double finalAns = 1 /(1 + exp(-dataContri - priorContri));
  return finalAns;
}

//' @title Performs a two-sample Bayesian nonparametric Polya tree test between two independent univariate samples
//'
//' @param x1  training sample from group 1
//' @param x0  training sample from group 0
//' @param cval     smoothing parameter
//' @param m        mean of centering Gaussian distribution
//' @param s        sd of centering Gaussian distribution
//' @param DOWN     Leave this number as 0
//' @param UP       Leave this number as 1
//' @param depth       Leave this number as 0
//' @param maxdepth    Maximum depth to truncate the tree. Default is 10.
//'
//' @return         log Bayes factor
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
// [[Rcpp::export]]
double TREECpp(const arma::vec& x1, const arma::vec& x0, const double& cval, const double& m, const double& s, const double& DOWN, const double& UP, const int& depth, const int& maxdepth = 10){

  const int N1 = x1.n_elem;
  const int N0 = x0.n_elem;
  const int N = N1 + N0;
  bool len0 = ((N1==0) && (N0==0)) && (N==0);
  if(len0 || (depth > maxdepth)){
    return 0;
  }
  else{

    int init_depth = 0;
    if(depth==0) init_depth = 1;
    double alpha = cval*pow(depth, 2) + init_depth;
    const double MIDDLE = (DOWN + UP)/2;
    const double b1 = R::qnorm(MIDDLE, m, s, true, false);
    const double b0 = R::qnorm(MIDDLE, m, s, true, false);
    uvec x1leftinds = find(x1 <= b1);
    uvec x0leftinds = find(x0 <= b0);
    arma::vec x1left = x1.elem( find(x1 <= b1) );
    arma::vec x0left = x0.elem( find(x0 <= b0) );
    arma::vec x1right = arma_setdiff(x1, x1left);
    arma::vec x0right = arma_setdiff(x0, x0left);
    int n1left = x1left.n_elem;
    int n1right = N1 - n1left;
    int n0left = x0left.n_elem;
    int n0right = N0 - n0left;
    int nleft = n1left + n0left;
    int nright = n1right + n0right;
    double ans = R::lbeta(alpha + nleft, alpha + nright) + R::lbeta(alpha, alpha) - R::lbeta(alpha + n1left, alpha+ n1right) - R::lbeta(alpha + n0left, alpha+ n0right);
    return (ans + TREECpp(x1left, x0left, cval, m, s, DOWN, MIDDLE, depth+1, maxdepth) + TREECpp(x1right, x0right, cval, m, s, MIDDLE, UP, depth+1, maxdepth));

  }

}


//' @title Fits the VNPDA model with a distinct smoothing parameter for each variable
//'
//' @param xtrain  Variables of training sample (matrix)
//' @param ytrain  Group labels of training sample
//' @param Xtest   Variables of testing sample for classification (matrix)
//' @param cval    Vector of smoothing parameter values
//' @param ay      Hyparameter 1 of Beta prior for rho_y
//' @param by      Hyparameter 2 of Beta prior for rho_y
//' @param ag      Hyparameter 1 of Beta prior for rho_gamma
//' @param bg      Hyparameter 2 of Beta prior for rho_gamma
//' @param maxdepth  Maximum depth to truncate the tree. Default is 10.
//'
//' @return         List of variable selection and classification results
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
// [[Rcpp::export]]
List VNPDA(const arma::mat& xtrain, const arma::vec& ytrain, const arma::mat& Xtest, const arma::vec& cval, const double& ay, const double& by, const double& ag, const double& bg, const int& maxdepth = 10) {

  const int P = xtrain.n_cols;
  const uvec x1inds = find(ytrain == 1);
  const uvec x0inds = find(ytrain == 0);
  const arma::mat xtrain1 = xtrain.rows(x1inds);
  const arma::mat xtrain0 = xtrain.rows(x0inds);
  const arma::vec manyones = ones<vec>(P);
  arma::vec omega = manyones / 2;
  arma::vec omegaOLD = omega;
  bool STOP = false;
  double requiredSum = 0.5*P;
  double omegaDIFF2sum = 0.0;
  double exponentANS;
  const arma::vec xbars = mean( xtrain , 0).t();
  const arma::vec SDs = stddev( xtrain , 0).t();
  while(!STOP){
    omegaDIFF2sum = 0.0;
    for(int j =0; j < P; j++){
      arma::vec geneX1 = sort(xtrain1.col(j));
      arma::vec geneX0 = sort(xtrain0.col(j));
      double mainRes = TREECpp(geneX1,geneX0,cval(j),xbars(j), SDs(j), 0, 1, 0, maxdepth);
      requiredSum -= omega(j);
      exponentANS = mainRes + log((bg + P - requiredSum - 1)/(ag + requiredSum ));
      omega(j) = 1/(1+exp(exponentANS));
      requiredSum += omega(j);
      omegaDIFF2sum += pow(omega(j) - omegaOLD(j),2);
      omegaOLD(j) = omega(j);
    }
    if(omegaDIFF2sum < 5){
      STOP = true;
    }
  }
  const int nTest = Xtest.n_rows;
  arma::vec ClassProb = zeros<vec>(nTest);
  arma::vec ClassPred = zeros<vec>(nTest);
  for(int i =0; i < nTest; i++){
    arma::vec newx = Xtest.row(i).t();
    ClassProb(i) = PredCpp(xtrain1, xtrain0, newx, cval, omega, ay, by, maxdepth);
    ClassPred(i) = round(ClassProb(i));
  }
  return Rcpp::List::create(Rcpp::Named("ClassPred") = ClassPred, Rcpp::Named("xi") = ClassProb, Rcpp::Named("omega") = omega);

}

//' @title Introduces random perturbation to a set of variables
//'
//' @param Z  matrix of values to be perturbed
//'
//' @return         perturbed matrix
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
// [[Rcpp::export]]
arma::mat PerturbCpp(arma::mat Z){
  const int n = Z.n_rows;
  const int P = Z.n_cols;
  for(int j =0; j<P; j++)
  {
    bool STOP = false;
    while(!STOP){
      STOP = true;
      for(int k=0; k<n; k++){
        for(int t = k+1; t<n; t++){
          if(Z(k,j)==Z(t,j)){
            STOP = false;
            Z(t,j) += R::runif(0.00001, 0.0001);
          }

        }

      }
    }

  }
  return Z;
}


//' @title Sampling with replacement from a vector of numbers
//'
//' @param x  Vector of elements to sample from
//' @param numObs  Number of draws required
//' @param weights  Sampling weights
//'
//' @return        vector of draws
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
// [[Rcpp::export]]
arma::vec ArmaSample(const arma::vec& x, const int& numObs, const arma::vec& weights){

  vec cuts = cumsum(weights);
  const vec U = randu(numObs);
  vec z(numObs);
  for(int i=0; i<numObs; i++){
    uvec pickedId = find(cuts <= U(i));
    z(i) = x(pickedId.n_elem);
  }
  return z;
}


//' @title Sampling from a mixture of univariate normals
//'
//' @param N        Number of rows in output
//' @param p        Number of columns in outout
//' @param indices  Recommended to be sequence from 1 to p
//' @param weights  Sampling probabilities
//' @param meanVec  Vector of Gaussian means
//' @param sdVec    Vector of Gaussian SDs
//'
//' @return        matrix N by p of draws
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
// [[Rcpp::export]]
arma::mat SampleMixNorm(const int& N, const int& p, const arma::vec indices, const arma::vec weights, const arma::vec meanVec, const arma::vec sdVec){

  arma::vec BigZ;
  arma::mat BigU(N,p, fill::randn);
  const int Components = weights.n_elem;

  for(int i =0; i<N; i++){

    BigZ = ArmaSample(indices, p, weights);
    for(int k = 0; k<Components; k++){

      uvec kInd = find(BigZ == indices(k));
      for(int j =0; j<kInd.n_elem; j++){
        BigU(i, kInd(j))*= sdVec(k);
        BigU(i, kInd(j))+= meanVec(k);
      }

    }
  }
  return BigU;
}

//' @title Calls the two-sample KS test function for two univariate samples in R
//'
//' @param x       Sample 1
//' @param y       Sample 0
//'
//' @return        KS test p-value
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
// [[Rcpp::export]]
double simpleKS(arma::colvec x, arma::colvec y) {
  Rcpp::Environment stats("package:stats");
  Rcpp::Function KSr = stats["ks.test"];
  List ans = KSr(Rcpp::_["x"] = x, Rcpp::_["y"]  = y, Rcpp::_["alternative"] = "two.sided", Rcpp::_["exact"] = R_NilValue);
  double ansFINAL = ans["p.value"];
  return ansFINAL;
}

//' @title Calls the two-sample KS test function for two multivariate samples in R
//'
//' @param X       Sample
//' @param y       Group labels (1 or 0)
//'
//' @return        KS test p-value
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
// [[Rcpp::export]]
arma::vec KSmatrix(arma::mat X, arma::colvec y) {

  const int P = X.n_cols;
  const uvec x1inds = find(y == 1);
  const uvec x0inds = find(y == 0);
  arma::mat xtrain1 = X.rows(x1inds);
  arma::mat xtrain0 = X.rows(x0inds);
  arma::vec pvalues(P);
  for(int j =0; j< P; j++){

    pvalues(j) = simpleKS(xtrain1.col(j), xtrain0.col(j));

  }
  return(pvalues);

}

//' @title Performs a two-sample Bayesian nonparametric Polya tree test between two independent univariate samples (for multiple candidate values of smoothing paramater)
//'
//' @param x1  training sample from group 1
//' @param x0  training sample from group 0
//' @param cval     candidate value for smoothing parameter (vector)
//' @param m        mean of centering Gaussian distribution
//' @param s        sd of centering Gaussian distribution
//' @param DOWN     Leave this number as 0
//' @param UP       Leave this number as 1
//' @param depth       Leave this number as 0
//' @param maxdepth    Maximum depth to truncate the tree. Default is 10.
//'
//' @return         log Bayes factor
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
// [[Rcpp::export]]
arma::vec TREECppCVselect(const arma::vec& x1, const arma::vec& x0, const arma::vec& cval, const double& m, const double& s, const double& DOWN = 0.0, const double& UP = 1.0, const int& depth =0.0, const int& maxdepth = 10){

  const int N1 = x1.n_elem;
  const int N0 = x0.n_elem;
  const int N = N1 + N0;
  bool len0 = ((N1==0) && (N0==0)) && (N==0);
  const int csize = cval.n_elem;
  if(len0 || (depth > maxdepth)){
    arma::vec ans(csize);
    ans.zeros();
    return ans;
  }
  else{

    arma::vec init_depth(csize);
    init_depth.zeros();
    if(depth==0) init_depth.ones();
    arma::vec alpha = cval*pow(depth, 2) + init_depth;
    const double MIDDLE = (DOWN + UP)/2;
    const double b1 = R::qnorm(MIDDLE, m, s, true, false);
    const double b0 = R::qnorm(MIDDLE, m, s, true, false);
    uvec x1leftinds = find(x1 <= b1);
    uvec x0leftinds = find(x0 <= b0);
    arma::vec x1left = x1.elem( find(x1 <= b1) );
    arma::vec x0left = x0.elem( find(x0 <= b0) );
    arma::vec x1right = arma_setdiff(x1, x1left);
    arma::vec x0right = arma_setdiff(x0, x0left);
    int n1left = x1left.n_elem;
    int n1right = N1 - n1left;
    int n0left = x0left.n_elem;
    int n0right = N0 - n0left;
    int nleft = n1left + n0left;
    int nright = n1right + n0right;
    arma::vec ans(csize);
    for(int k =0; k<csize; k++){

      ans(k) = R::lbeta(alpha(k) + nleft, alpha(k) + nright) + R::lbeta(alpha(k), alpha(k)) - R::lbeta(alpha(k) + n1left, alpha(k)+ n1right) - R::lbeta(alpha(k) + n0left, alpha(k)+ n0right);

    }
    return (ans + TREECppCVselect(x1left, x0left, cval, m, s, DOWN, MIDDLE, depth+1, maxdepth) + TREECppCVselect(x1right, x0right, cval, m, s, MIDDLE, UP, depth+1, maxdepth));

  }

}

//' @title Calculate classification probabilities for binary Polya tree classification problem with a single feature (for multiple candidate values of smoothing parameter)
//'
//' @param x1       training sample from group 1
//' @param x0       training sample from group 0
//' @param xnew     new data point
//' @param cval     candidate values of smoothing parameter
//' @param m        mean of centering Gaussian distribution
//' @param s        sd of centering Gaussian distribution
//' @param DOWN     Leave this number as 0
//' @param UP       Leave this number as 1
//' @param depth       Leave this number as 0
//' @param maxdepth    Maximum depth to truncate the tree. Default is 10.
//'
//' @return         Classification probability
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
//' @examples
//' samp1 <- rnorm(50,4,1)
//' samp0 <- rt(40,3)
//' xtest <- rt(1,3)
//' DividePredCpp(x1 = samp1, x0 = samp0, xnew = xtest, cval = 1, m = mean(c(samp1,samp0)), s = sd(c(samp1,samp0)))
arma::vec DividePredCppCVselect(const arma::vec& x1, const arma::vec& x0, const double& xnew, const arma::vec& cval, const double& m, const double& s, const double& DOWN=0.0, const double& UP=1.0, const int& depth = 0, const int& maxdepth = 10) {

  const int n1 = x1.n_elem;
  const int n0 = x0.n_elem;
  const int csize = cval.n_elem;
  if( (n1+n0 == 0) || (depth > maxdepth) ){
    arma::vec ans(csize);
    ans.zeros();
    return ans;
  }
  else
  {
    arma::vec init_depth(csize);
    init_depth.zeros();
    if(depth==0) init_depth.ones();
    arma::vec alpha = cval*pow(depth, 2) + init_depth;
    const double MIDDLE = (DOWN+UP)/2;
    const double b = R::qnorm(MIDDLE, m, s, TRUE, FALSE);
    uvec x1leftinds = find(x1 <= b);
    uvec x0leftinds = find(x0 <= b);
    const arma::vec x1left = x1.elem( find(x1 <= b) );
    const arma::vec x0left = x0.elem( find(x0 <= b) );
    const arma::vec x1right = arma_setdiff(x1, x1left);
    const arma::vec x0right = arma_setdiff(x0, x0left);
    bool left = xnew <= b;
    int n1left = x1left.n_elem;
    int n0left = x0left.n_elem;
    int n1right = n1 - n1left;
    int n0right = n0 - n0left;
    arma::vec nextx1;
    arma::vec nextx0;
    arma::vec ans(csize);
    for(int k =0; k<csize; k++){

      ans(k) = left*log(alpha(k) + n1left) - left*log(alpha(k) + n0left) + (1-left)*log(alpha(k) + n1right) - (1-left)*log(alpha(k) + n0right) - log(2*alpha(k) + n1) + log(2*alpha(k) + n0);

    }

    if(left)
    {
      nextx1 = x1left;
      nextx0 = x0left;
    }
    else
    {
      nextx1 = x1right;
      nextx0 = x0right;
    }
    return (ans + DividePredCppCVselect(nextx1,nextx0,xnew,cval,m,s,left*DOWN+(1-left)*MIDDLE,left*MIDDLE+(1-left)*UP,depth+1,maxdepth) );
  }
}

//' @title Calculate classification probabilities for binary Polya tree classification problem with multiple features (for mutiple candidate values of smoothing parameter)
//'
//' @param xtrain1  training sample from group 1 (matrix)
//' @param xtrain0  training sample from group 0 (matrix)
//' @param xtest    a single new data point (vector)
//' @param cval     candidate values for smoothing parameter (matrix)
//' @param omega    vector of variable selection weights
//' @param ay        hyperparameter 1 of beta prior
//' @param by        hyperparameter 2 of beta prior
//' @param maxdepth    Maximum depth to truncate the tree. Default is 10.
//'
//' @return         Classification probability
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
// [[Rcpp::export]]
arma::vec PredCppCVselect(arma::mat xtrain1, arma::mat xtrain0, arma::vec xtest, const arma::mat& cval, const arma::mat& omega, double ay, double by, int maxdepth = 10) {

  int P = xtest.n_elem;
  int N1 = xtrain1.n_rows;
  int N0 = xtrain0.n_rows;
  arma::vec logContribution = zeros<vec>(P);
  arma::mat xtestMAT = zeros<mat>(1,P);
  xtestMAT.row(0) = xtest.t();
  arma::mat tempMat = join_cols(join_cols(xtrain1, xtrain0), xtestMAT);
  const vec BigMeans = mean( tempMat, 0).t();
  const vec BigSDs = stddev( tempMat , 0).t();

  const int numCombs = cval.n_rows;
  arma::vec finalAns(numCombs);
  finalAns.zeros();
  arma::vec anss(numCombs);
  arma::mat results(numCombs,P);

  for(int j =0; j < P; j++){
    arma::vec x1 = xtrain1.col(j);
    arma::vec x0 = xtrain0.col(j);
    anss = DividePredCppCVselect(x1, x0, xtestMAT(0,j), cval.col(j), BigMeans(j), BigSDs(j), 0, 1, 0, maxdepth);
    results.col(j) = anss % omega.col(j);
  }

  arma::vec dataContri = sum(results, 1);
  double priorContri = log(ay + N1) - log(by + N0);
  for(int k =0; k<numCombs; k++){

    finalAns(k) = 1 /(1 + exp(-dataContri(k) - priorContri));

  }

  return(finalAns);
}

//' @title Fits the VNPDA model for a set of candidate value of smoothing parameters for each variable and returns the optimal smoothing parameter value
//'
//' @param xtrain  Variables of training sample (matrix)
//' @param ytrain  Group labels of training sample
//' @param Xtest   Variables of testing sample for classification (matrix)
//' @param cval     candidate values for smoothing parameter (matrix)
//' @param ay      Hyparameter 1 of Beta prior for rho_y
//' @param by      Hyparameter 2 of Beta prior for rho_y
//' @param ag      Hyparameter 1 of Beta prior for rho_gamma
//' @param bg      Hyparameter 2 of Beta prior for rho_gamma
//' @param maxdepth  Maximum depth to truncate the tree. Default is 10.
//'
//' @return         Optimal smoothing parameter settings
//'
//' @export
//' @rawNamespace useDynLib(VaDA)
//'
// [[Rcpp::export]]
List CVselectC(const arma::mat& xtrain, const arma::vec& ytrain, const arma::mat& cval, const double& ay, const double& by, const double& ag, const double& bg, const int& maxdepth = 10) {

  const int P = xtrain.n_cols;
  const int numCombs = cval.n_rows;
  const uvec x1inds = find(ytrain == 1);
  const uvec x0inds = find(ytrain == 0);
  const arma::mat xtrain1 = xtrain.rows(x1inds);
  const arma::mat xtrain0 = xtrain.rows(x0inds);
  const arma::mat manyones = ones<vec>(P);
  arma::mat omega(numCombs, P);
  omega.fill(0.5);
  arma::mat omegaOLD = omega;
  bool STOP = false;
  arma::vec requiredSum(numCombs);
  requiredSum.fill(0.5*P);
  arma::vec omegaDIFF2sum(numCombs);
  omegaDIFF2sum.zeros();
  arma::vec exponentANS(numCombs);
  const arma::vec xbars = mean( xtrain , 0).t();
  const arma::vec SDs = stddev( xtrain , 0).t();
  while(!STOP){
    omegaDIFF2sum.zeros();
    for(int j =0; j < P; j++){
      arma::vec geneX1 = sort(xtrain1.col(j));
      arma::vec geneX0 = sort(xtrain0.col(j));
      arma::vec mainRes = TREECppCVselect(geneX1,geneX0,cval.col(j),xbars(j), SDs(j), 0, 1, 0, maxdepth);
      //cout << j << endl;
      requiredSum = requiredSum - omega.col(j);
      for(int k=0; k< numCombs; k++){

        exponentANS(k) = mainRes(k) + log((bg + P - requiredSum(k) - 1)/(ag + requiredSum(k) ));
        omega(k,j) = 1/(1+exp(exponentANS(k)));

      }
      requiredSum = requiredSum + omega.col(j);
      omegaDIFF2sum = omegaDIFF2sum + (omega.col(j) - omegaOLD.col(j))%(omega.col(j) - omegaOLD.col(j));
      omegaOLD.col(j) = omega.col(j);
    }
    if(omegaDIFF2sum.max() < 5){
      STOP = true;
      //cout << "STOP" << endl;
    }
  }
  const int nTrain = xtrain.n_rows;
  arma::mat ClassProb(numCombs, nTrain);
  arma::mat ClassPred(numCombs, nTrain);
  ClassProb.zeros();
  ClassPred.zeros();
  arma::vec ErrorsVec(numCombs);
  ErrorsVec.zeros();
  arma::vec true_response(numCombs);
  for(int i =0; i < nTrain; i++){
    arma::vec newx = xtrain.row(i).t();
    ClassProb.col(i) = PredCppCVselect(xtrain1, xtrain0, newx, cval, omega, ay, by, maxdepth);
    //cout << "Finished observation" << i+1 << " \n";
    ClassPred.col(i) = round(ClassProb.col(i));
    true_response.fill(ytrain(i));
    ErrorsVec = ErrorsVec + (true_response - ClassPred.col(i))%(true_response - ClassPred.col(i));
  }
  arma::vec BestC = cval.row(ErrorsVec.index_min()).t();
  return Rcpp::List::create(Rcpp::Named("Errors") = ErrorsVec, Rcpp::Named("BestC") = BestC);;

}

