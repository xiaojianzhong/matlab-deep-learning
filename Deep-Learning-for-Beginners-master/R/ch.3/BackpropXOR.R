source("./Sigmoid.R")

BackpropXOR <- function(W1, W2, X, D) {
  alpha <- 0.9

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
    W1 <- W1 + dW1

    dW2 <- alpha * delta %*% t(y1)
    W2 <- W2 + dW2
  }
  return(list("W1"=W1, "W2"=W2))
}
