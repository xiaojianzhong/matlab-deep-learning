Sigmoid <- function(x) {
  y <- 1 / (1 + exp(-x))
  return(y)
}
