rm(list=ls())

source("./DeltaSGD.R")

X <- matrix(c(
  0, 0, 1,
  0, 1, 1,
  1, 0, 1,
  1, 1, 1
), nrow=4, ncol=3, byrow=TRUE)

D <- c(
  0,
  0,
  1,
  1
)

W <- array(runif(3, min=-1, max=1), c(1, 3))

for (epoch in 1:10000) {
  W <- DeltaSGD(W, X, D)
}

N <- 4
for (k in 1:N) {
  x <- X[k,]
  v <- W %*% x
  y <- Sigmoid(v)
  print(y)
}
