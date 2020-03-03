rng(3)

X <- list(
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
)

D <- matrix(c(
  1, 0, 0, 0, 0,
  0, 1, 0, 0, 0,
  0, 0, 1, 0, 0,
  0, 0, 0, 1, 0,
  0, 0, 0, 0, 1
), nrow=5, ncol=5, byrow=TRUE)

W1 <- matrix(runif(1250, min=-1, max=1), nrow=50, ncol=25)
W2 <- matrix(runif(250, min=-1, max=1), nrow=5, ncol=50)

for (epoch in 1:10000) { # train
  Ws <- MultiClass(W1, W2, X, D)
  W1 <- Ws$W1
  W2 <- Ws$W2
}

N <- 5 # inference
for (k in 1:N) {
  x <- as.vector(t(X[[k]]))
  v1 <- W1 %*% x
  y1 <- Sigmoid(v1)
  v <- W2 %*% y1
  y <- Softmax(v)
  print(y)
}
