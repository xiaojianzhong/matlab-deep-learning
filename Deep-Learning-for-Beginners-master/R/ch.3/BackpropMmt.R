source('./Sigmoid.R')

BackpropMmt <- function(W1, W2, X, D) {
  alpha <- 0.9
  beta <- 0.9

  mmt1 <- matrix(0, nrow=dim(W1)[1], ncol=dim(W1)[2])
  mmt2 <- matrix(0, nrow=dim(W2)[1], ncol=dim(W2)[2])

  N <- 4
  for (k in 1:N) {
    x <- X[k,]
    d <- D[k]

    v1 <- W1 %*% x
    y1 <- Sigmoid(v1)
    v <- W2 %*% y1
    y <- Sigmoid(v)

    e <- d - y
    delta <- y * (1-y) * e

    e1 <- t(W2) %*% delta
    delta1 <- y1 * (1-y1) * e1

    dW1 <- alpha * delta1 %*% t(x)
    mmt1 <- dW1 + beta * mmt1
    W1 <- W1 + mmt1

    dW2 <- alpha * delta %*% t(y1)
    mmt2 <- dW2 + beta * mmt2
    W2 <- W2 + mmt2
  }
  return(list("W1"=W1, "W2"=W2))
}
