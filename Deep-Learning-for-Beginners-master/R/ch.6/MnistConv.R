source("./Conv.R")
source("./ReLU.R")
source("./Pool.R")
source("./Softmax.R")
source("./Sub2ind.R")
source("./conv2.R")

MnistConv <- function(W1, W5, Wo, X, D) {
  alpha <- 0.01
  beta <- 0.95

  momentum1 <- array(0, dim(W1))
  momentum5 <- array(0, dim(W5))
  momentumo <- array(0, dim(Wo))

  N <- length(D)

  bsize <- 100
  blist <- seq(1, N-bsize+1, bsize)

  # One epoch loop
  #
  for (batch in 1:length(blist)) {
    dW1 <- array(0, dim(W1))
    dW5 <- array(0, dim(W5))
    dWo <- array(0, dim(Wo))

    # Mini-batch loop
    #
    begin <- blist[batch]
    for (k in begin:(begin+bsize-1)) {
      # Forward pass = inference
      #
      x <- X[, , k]                               # Input,       28x28
      y1 <- Conv(x, W1)                           # Convolution, 20x20x20
      y2 <- ReLU(y1)                              #
      y3 <- Pool(y2)                              # Pool,        10x10x20
      y4 <- as.vector(y3)                         #              2000
      v5 <- W5 %*% y4                             # ReLU,        100
      y5 <- ReLU(v5)                              #
      v <- Wo %*% y5                              # Softmax,     10
      y <- Softmax(v)                             #

      # One-hot encoding
      #
      d <- array(0, c(10, 1))
      d[sub2ind(dim(d), D[k], 1)] <- 1

      # Backpropagation
      #
      e <- d - y                        # Output layer
      delta <- e

      e5 <- t(Wo) %*% delta             # Hidden(ReLU) layer
      delta5 <- (y5 > 0) * e5

      e4 <- t(W5) %*% delta5            # Pooling layer

      e3 <- array(e4, dim(y3))

      e2 <- array(0, dim(y2))
      W3 <- array(1, dim(y2)) / (2*2)
      for (c in 1:20) {
        e2[, , c] <- kronecker(e3[, , c], array(1, c(2, 2))) * W3[, , c]
      }

      delta2 <- (y2 > 0) * e2           # ReLU layer

      delta1_x <- array(0, dim(W1)) # Convolution layer
      for (c in 1:20) {
        delta1_x[, , c] <- conv2(x, delta2[, , c])
      }

      dW1 <- dW1 + delta1_x
      dW5 <- dW5 + delta5 %*% t(y4)
      dWo <- dWo + delta %*% t(y5)
    }

    # Update weights
    #
    dW1 <- dW1 / bsize
    dW5 <- dW5 / bsize
    dWo <- dWo / bsize

    momentum1 <- alpha * dW1 + beta * momentum1
    W1 <- W1 + momentum1

    momentum5 <- alpha * dW5 + beta * momentum5
    W5 <- W5 + momentum5

    momentumo <- alpha * dWo + beta * momentumo
    Wo <- Wo + momentumo
  }
  return(list("W1"=W1, "W5"=W5, "Wo"=Wo))
}
