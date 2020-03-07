rm(list=ls())

source("./DeltaSGD.R")
source("./DeltaBatch.R")
source("./Sigmoid.R")

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

E1 <- array(0, c(1000, 1))
E2 <- array(0, c(1000, 1))

W1 <- matrix(runif(3, min=-1, max=1), nrow=1, ncol=3)
W2 <- W1

for (epoch in 1:1000) { # train
  W1 <- DeltaSGD(W1, X, D)
  W2 <- DeltaBatch(W2, X, D)

  es1 <- 0
  es2 <- 0
  N <- 4
  for (k in 1:N) {
    x <- X[k,]
    d <- D[k]

    v1 <- W1 %*% x
    y1 <- Sigmoid(v1)
    es1 <- es1 + (d - y1)^2

    v2 <- W2 %*% x
    y2 <- Sigmoid(v2)
    es2 <- es2 + (d - y2)^2
  }
  E1[epoch] <- es1 / N
  E2[epoch] <- es2 / N
}

plot(x=c(), y=c(), xlab="Epoch", ylab="Average of Training error", xlim=c(0, 1000), ylim=c(0, 1))
lines(1:1000, E1, col="red")
lines(1:1000, E2, col="blue", lty="dotted")
legend(x=0, y=1, legend=c("SGD", "Batch"), col=c("red", "blue"), lty=c("solid", "dotted"))
