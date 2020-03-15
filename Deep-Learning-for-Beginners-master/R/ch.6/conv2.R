library(OpenImageR)

conv2 <- function(x, filter, step=1) {
  dimf <- dim(filter)
  frow <- dimf[1]
  fcol <- dimf[2]
  dimx <- dim(x)
  xrow <- dimx[1]
  xcol <- dimx[2]

  yrow <- xrow - frow + 1
  ycol <- xcol - fcol + 1

  rowOffset <- floor((frow - 1) / 2)
  colOffset <- floor((fcol - 1) / 2)

  y <- convolution(x, filter)[seq(rowOffset+1,rowOffset+yrow,step), seq(colOffset+1,colOffset+ycol,step)]
  return(y)
}
