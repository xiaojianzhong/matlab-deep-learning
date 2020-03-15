source("./conv2.R")

Conv <- function(x, W) {
  dimW <- dim(W)
  wrow <- dimW[1]
  wcol <- dimW[2]
  numFilters <- dimW[3]
  dimx <- dim(x)
  xrow <- dimx[1]
  xcol <- dimx[2]

  yrow <- xrow - wrow + 1
  ycol <- xcol - wcol + 1

  y <- array(0, c(yrow, ycol, numFilters))

  for (k in 1:numFilters) {
    filter <- W[, , k]
    y[, , k] <- conv2(x, filter)
  }
  return(y)
}
