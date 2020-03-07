rm(list=ls())

source("./rng.R")
source("./MultiClass.R")
source("./Sigmoid.R")
source("./Softmax.R")

rng(3)

X <- array(
  cbind(
    matrix(c(
      0, 1, 1, 0, 0,
      0, 0, 1, 0, 0,
      0, 0, 1, 0, 0,
      0, 0, 1, 0, 0,
      0, 1, 1, 1, 0
    ), nrow=5, ncol=5, byrow=TRUE),
    matrix(c(
      1, 1, 1, 1, 0,
      0, 0, 0, 0, 1,
      0, 1, 1, 1, 0,
      1, 0, 0, 0, 0,
      1, 1, 1, 1, 1
    ), nrow=5, ncol=5, byrow=TRUE),
    matrix(c(
      1, 1, 1, 1, 0,
      0, 0, 0, 0, 1,
      0, 1, 1, 1, 0,
      0, 0, 0, 0, 1,
      1, 1, 1, 1, 0
    ), nrow=5, ncol=5, byrow=TRUE),
    matrix(c(
      0, 0, 0, 1, 0,
      0, 0, 1, 1, 0,
      0, 1, 0, 1, 0,
      1, 1, 1, 1, 1,
      0, 0, 0, 1, 0
    ), nrow=5, ncol=5, byrow=TRUE),
    matrix(c(
      1, 1, 1, 1, 1,
      1, 0, 0, 0, 0,
      1, 1, 1, 1, 0,
      0, 0, 1, 0, 1,
      1, 1, 1, 1, 0
    ), nrow=5, ncol=5, byrow=TRUE)
  ),
  dim=c(5, 5, 5)
)

D <- matrix(c(
  1, 0, 0, 0, 0,
  0, 1, 0, 0, 0,
  0, 0, 1, 0, 0,
  0, 0, 0, 1, 0,
  0, 0, 0, 0, 1
), nrow=5, ncol=5, byrow=TRUE)

W1 <- array(runif(1250, min=-1, max=1), c(50, 25))
W2 <- array(runif(250, min=-1, max=1), c(5, 50))

for (epoch in 1:10000) { # train
  Ws <- MultiClass(W1, W2, X, D)
  W1 <- Ws$W1
  W2 <- Ws$W2
}

N <- 5 # inference
for (k in 1:N) {
  x <- as.vector(t(X[, , k]))
  v1 <- W1 %*% x
  y1 <- Sigmoid(v1)
  v <- W2 %*% y1
  y <- Softmax(v)
  print(y)
}
