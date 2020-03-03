ReLU <- function (x) {
  y <- vector(mode="numeric", length=length(x))
  for (i in 1:length(x)) {
    y[i] <- max(0, x[i])
  }
  return(y)
}
