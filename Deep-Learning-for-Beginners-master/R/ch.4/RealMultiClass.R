X <- list(
  matrix(c(
    0, 0, 1, 1, 0,
    0, 0, 1, 1, 0,
    0, 1, 0, 1, 0,
    0, 0, 0, 1, 0,
    0, 1, 1, 1, 0
  ), nrow=5, ncol=5, byrow=TRUE),
  matrix(c(
    1, 1, 1, 1, 0,
    0, 0, 0, 0, 1,
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 1, 1, 1, 1
  ), nrow=5, ncol=5, byrow=TRUE),
  matrix(c(
    1, 1, 1, 1, 0,
    0, 0, 0, 0, 1,
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 1, 1, 1, 0
  ), nrow=5, ncol=5, byrow=TRUE),
  matrix(c(
    0, 1, 1, 1, 0,
    0, 1, 0, 0, 0,
    0, 1, 1, 1, 0,
    0, 0, 0, 1, 0,
    0, 1, 1, 1, 0
  ), nrow=5, ncol=5, byrow=TRUE),
  matrix(c(
    0, 1, 1, 1, 1,
    0, 1, 0, 0, 0,
    0, 1, 1, 1, 0,
    0, 0, 0, 1, 0,
    1, 1, 1, 1, 0
  ), nrow=5, ncol=5, byrow=TRUE)
)

N <- 5 # inference
for (k in 1:N) {
  x <- as.vector(t(X[[k]]))
  v1 <- W1 %*% x
  y1 <- Sigmoid(v1)
  v <- W2 %*% y1
  y <- Softmax(v)
  print(y)
}
