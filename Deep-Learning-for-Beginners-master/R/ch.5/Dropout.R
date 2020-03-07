Dropout <- function(y, ratio) {
  d <- dim(y)
  m <- d[1]
  n <- d[2]
  ym <- array(0, d)

  num <- round(m * n * (1 - ratio))
  idx <- sample(m * n, num)
  ym[idx] <- 1 / (1 - ratio)
  return(ym)
}
