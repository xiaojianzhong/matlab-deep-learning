source("./conv2.R")

# 2 x 2 mean pooling
#
Pool <- function(x) {
  dimx <- dim(x)
  xrow <- dimx[1]
  xcol <- dimx[2]
  numFilters <- dimx[3]

  y <- array(0, c(xrow/2, xcol/2, numFilters))
  for (k in 1:numFilters) {
    filter <- array(1, c(2, 2)) / (2*2) # for mean
    y[, , k] <- conv2(x[, , k], filter, step=2)
  }
  return(y)
}
