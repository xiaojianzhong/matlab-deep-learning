source('./DeepDropout.R')
source('./Sigmoid.R')
source('./Softmax.R')

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

W1 <- matrix(runif(500, min=-1, max=1), nrow=20, ncol=25)
W2 <- matrix(runif(400, min=-1, max=1), nrow=20, ncol=20)
W3 <- matrix(runif(400, min=-1, max=1), nrow=20, ncol=20)
W4 <- matrix(runif(100, min=-1, max=1), nrow=5, ncol=20)

for (epoch in 1:10000) { # train
  Ws <- DeepDropout(W1, W2, W3, W4, X, D)
  W1 <- Ws$W1
  W2 <- Ws$W2
  W3 <- Ws$W3
  W4 <- Ws$W4
}

N <- 5 # inference
for (k in 1:N) {
  x <- as.vector(t(X[, , k]))
  v1 <- W1 %*% x
  y1 <- Sigmoid(v1)

  v2 <- W2 %*% y1
  y2 <- Sigmoid(v2)

  v3 <- W3 %*% y2
  y3 <- Sigmoid(v3)

  v <- W4 %*% y3
  y <- Softmax(v)
  print(y)
}
