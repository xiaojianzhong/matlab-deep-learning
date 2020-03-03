source('./Sigmoid.R')

DeltaXOR <- function(W, X, D) {
  alpha <- 0.9

  N <- 4
  for (k in 1:N) {
    x <- X[k,]
    d <- D[k]

    v <- W %*% x
    y <- Sigmoid(v)

    e <- d - y
    delta <- y * (1-y) * e

    dW <- alpha * delta * x # delta rule

    W <- W + dW
  }

  return(W)
}
