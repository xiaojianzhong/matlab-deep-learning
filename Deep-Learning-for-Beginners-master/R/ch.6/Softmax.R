Softmax <- function(x) {
  ex <- exp(x)
  y <- ex / sum(ex)
  return(y)
}
