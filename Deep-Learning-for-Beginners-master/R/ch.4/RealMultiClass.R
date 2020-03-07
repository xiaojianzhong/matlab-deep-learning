rm(list=ls())

source("./TestMultiClass.R")
source("./Sigmoid.R")
source("./Softmax.R")

X <- array(0, c(5, 5, 5))

X[, , 1] <- matrix(c(
  0, 0, 1, 1, 0,
  0, 0, 1, 1, 0,
  0, 1, 0, 1, 0,
  0, 0, 0, 1, 0,
  0, 1, 1, 1, 0
), nrow=5, ncol=5, byrow=TRUE)

X[, , 2] <- matrix(c(
  1, 1, 1, 1, 0,
  0, 0, 0, 0, 1,
  0, 1, 1, 1, 0,
  1, 0, 0, 0, 1,
  1, 1, 1, 1, 1
), nrow=5, ncol=5, byrow=TRUE)

X[, , 3] <- matrix(c(
  1, 1, 1, 1, 0,
  0, 0, 0, 0, 1,
  0, 1, 1, 1, 0,
  1, 0, 0, 0, 1,
  1, 1, 1, 1, 0
), nrow=5, ncol=5, byrow=TRUE)

X[, , 4] <- matrix(c(
  0, 1, 1, 1, 0,
  0, 1, 0, 0, 0,
  0, 1, 1, 1, 0,
  0, 0, 0, 1, 0,
  0, 1, 1, 1, 0
), nrow=5, ncol=5, byrow=TRUE)

X[, , 5] <- matrix(c(
  0, 1, 1, 1, 1,
  0, 1, 0, 0, 0,
  0, 1, 1, 1, 0,
  0, 0, 0, 1, 0,
  1, 1, 1, 1, 0
), nrow=5, ncol=5, byrow=TRUE)

N <- 5 # inference
for (k in 1:N) {
  x <- as.vector(X[, , k])
  v1 <- W1 %*% x
  y1 <- Sigmoid(v1)
  v <- W2 %*% y1
  y <- Softmax(v)
  print(y)
}
