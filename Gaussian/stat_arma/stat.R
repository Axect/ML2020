set.seed(42)

library("Rcpp")
library("RcppArmadillo")
sourceCpp("arma_stat.cpp")

x <- matrix(rnorm(4 * 4), 4, 4)
z <- x %*% t(x)

getEigenValues(x)
