
#'
#' @import Rcpp
#' @import ggplot2
#' @import RcppAlgos
#' @import RcppArmadillo
#' @import stats
#'

#' @title G inverse calculation
#'
#' Describe
#'
#' @param p Numeric vector of probabilities
#' @param xbar Numeric Gaussian mean
#' @param s Numeric Gaussian sd
#'
#' @return Numeric vector
#'
#' @examples
#' q_vec <- c(1:5)/6
#' Ginv(q_vec,xbar=0,s=1)
#'
#' @export
Ginv <- function(p,xbar,s){
return(qnorm(p,mean=xbar,sd=s))
}

#' @title Calculate recursive histograms
#'
#'
#' @param x1 Numeric vector for sample 1
#' @param x0 Numeric vector for sample 2
#' @param prev_n1 Numeric
#' @param prev_n0 Numeric
#' @param c Numeric
#' @param DOWN Numeric
#' @param UP Numeric
#' @param m1 Numeric
#' @param s1 Numeric
#' @param m0 Numeric
#' @param s0 Numeric
#' @param depth Numeric
#' @param maxdepth Numeric
#' @param samples Numeric for number of samples
#'
#' @return a list object
#' @export
#'
#' @examples
#' test_samp <- rnorm(60,0,3)
#' HistoTree3(x1=test_samp,DOWN=0,c=3,UP=1,m1=mean(test_samp),s1=sd(test_samp))
HistoTree3<-function(x1,x0=NULL,prev_n1=0,prev_n0=0,c,DOWN,UP,m1,s1,m0=NULL,s0=NULL,depth=0,maxdepth=20,samples=1)
{
	if(depth > maxdepth)
	{
		return(0)
	}
	else
	{
		n1 <- length(x1)
		MIDDLE <- (DOWN+UP)/2
		b1 <- Ginv(MIDDLE,m1,s1)
		x1leftInds <- which(x1 < b1)
		x1rightInds <- which(x1 >= b1)
		x1left <- x1[x1leftInds]
		x1right <- x1[x1rightInds]
		if(samples == 2)
		{
			n0 <- length(x0)
			b0 <- Ginv(MIDDLE,m0,s0)
			x0leftInds <- which(x0 < b0)
			x0rightInds <- which(x0 >= b0)
			x0left <- x0[x0leftInds]
			x0right <- x0[x0rightInds]
		}
		if(depth==0)
		{
			ans1 <- 0
			ans0 <- 0
		}
		else
		{
			alpha <- c*(depth-1)^2 + as.numeric(depth==1)
			if(samples==2)
			{
				ans1 <- log(alpha + n1) - log(2*alpha + prev_n1)
				ans0 <- log(alpha + n0) - log(2*alpha + prev_n0)
			}
			else
			{
				ans1 <- log(alpha + n1) - log(2*alpha + prev_n1)
				ans0 <- 0
			}
		}
		if(depth==maxdepth)
		{
			return(list(ans1,ans0))
		}
		else
		{
			if(samples==2)
			{
				RES1 <- HistoTree3(x1=x1left,x0=x0left,prev_n1=n1,prev_n0=n0,c,DOWN,MIDDLE,m1,s1,m0,s0,depth+1,maxdepth,samples)
				RES0 <- HistoTree3(x1=x1right,x0=x0right,prev_n1=n1,prev_n0=n0,c,MIDDLE,UP,m1,s1,m0,s0,depth+1,maxdepth,samples)
				return( list(c(ans1 + RES1[[1]], ans1+ RES0[[1]]),c(ans0 + RES1[[2]], ans0 + RES0[[2]]))  )
			}
			else
			{
				return( list(c(ans1 + HistoTree3(x1=x1left,x0=x0left,prev_n1=n1,prev_n0=n0,c,DOWN,MIDDLE,m1,s1,m0,s0,depth+1,maxdepth,samples)[[1]], ans1+ HistoTree3(x1=x1right,x0=x0right,prev_n1=n1,prev_n0=n0,c,MIDDLE,UP,m1,s1,m0,s0,depth+1,maxdepth,samples)[[1]]), 0)  )
			}
		}

	}
}
#'

#' @title Plots Polya Tree Predictive Density
#'
#' @param x1 Numeric vector sample 1
#' @param x0 Numeric vector sample 1
#' @param c Numeric smoothing parameter
#' @param maxdepth Numeric maximum tree depth
#' @param mainTitle Text for plot heading
#' @param HorizLab Text for x-axis label
#' @param x_limits Limits of x-axis
#'
#' @return ggplot object
#'
#' @export
#'
#' @examples
#' x_samp <- rnorm(30,3,2)
#' y_samp <- rt(30,2)
#' PlotPT(x1=x_samp, x0 = y_samp, c=2)
PlotPT<-function(x1,x0=NULL,c,maxdepth=10,mainTitle=NULL,HorizLab=NULL,x_limits=NULL)
{
	samples <- 1 + as.numeric(length(x0)>0)
	m <- mean(c(x1,x0))
	s <- sd(c(x1,x0))
	NumOfBins <- 2^maxdepth
	logFINAL <- HistoTree3(x1=x1,x0=x0,prev_n1=0,prev_n0=0,c,DOWN=0,UP=1,m1=m,s1=s,m0=m,s0=s,depth=0,maxdepth=maxdepth,samples=samples)
	FINAL <- lapply(logFINAL,function(a) {exp(a)})
	cutOFFs <- Ginv(seq(0,1,(1/NumOfBins)),mean(c(x1,x0)),sd(c(x1,x0)))
	if(is.null(x_limits)){

		x_limits = c(min(x1) - 0.2*s, max(x1) + 0.2*s)

	}
  #browser()
	Plot1<-ggplot(data.frame(x = c(x1,x0)), aes(x = x)) + stat_function(fun = PT_Pred_Density, args=list(probs=FINAL[[1]],cutOFFs=cutOFFs,m=m,s=s), geom="line", colour="#001dff") + stat_function(fun = PT_Pred_Density, args = list(probs=FINAL[[1]],cutOFFs=cutOFFs,m=m,s=s),geom = "area", fill = "#001dff", alpha = .3) + scale_x_continuous(limits=x_limits)+ geom_rug(aes(x = x, y=0), sides="b", colour=c(rep("#001dff",length(x1)), rep("#ff0077",length(x0))) )
	if(!is.null(mainTitle)){

		Plot1 <- Plot1 + ggtitle(mainTitle)

	}
	if(is.null(x0)){

		Plot1 <- Plot1

	}
	else{

		Plot1<- Plot1 + stat_function(fun = PT_Pred_Density, args=list(probs=FINAL[[2]],cutOFFs=cutOFFs,m=m,s=s),geom="line", colour="#ff0077")+ stat_function(fun = PT_Pred_Density, args = list(probs=FINAL[[2]],cutOFFs=cutOFFs,m=m,s=s),geom = "area", fill = "#ff0077", alpha = .3)


	}
	print(Plot1)
	return(Plot1)

}
#'

#' @title Calculates Polya Tree Predictive Density
#'
#' @param x Numeric vector of sample to be plotted
#' @param probs Numeric vector of probabilities
#' @param cutOFFs Bin cutoffs
#' @param m Numeric mean of centering distribution
#' @param s Numeric sd of centering distribution
#'
#' @return Numeric vector of predictive density
#'
#' @export
#'
PT_Pred_Density<-function(x,probs,cutOFFs,m=0,s=1){

	ind <- unlist(lapply(x,function(b){ which.min(b > cutOFFs)  }))
	ans <- probs[ind-1] * length(probs) * dnorm(x,m,s)
	#browser()
	#ans <- dnorm(x,m,s)
	return(ans)

}

Sys.setenv("PKG_CXXFLAGS"="-std=c++11")
