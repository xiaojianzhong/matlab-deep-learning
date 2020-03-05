source("./BackpropCE.R")
source("./BackpropXOR.R")
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

E1 <- vector(mode="numeric", length=1000)
E2 <- vector(mode="numeric", length=1000)

W11 <- matrix(runif(12, min=-1, max=1), nrow=4, ncol=3) # Cross entropy
W12 <- matrix(runif(4, min=-1, max=1), nrow=1, ncol=4)
W21 <- W11 # Sum of squared error
W22 <- W12

for (epoch in 1:1000) {
  W1s <- BackpropCE(W11, W12, X, D)
  W11 <- W1s$W1
  W12 <- W1s$W2
  W2s <- BackpropXOR(W21, W22, X, D)
  W21 <- W2s$W1
  W22 <- W2s$W2

  es1 <- 0
  es2 <- 0
  N <- 4
  for (k in 1:N) {
    x <- X[k,]
    d <- D[k]

    v1 <- W11 %*% x
    y1 <- Sigmoid(v1)
    v <- W12 %*% y1
    y <- Sigmoid(v)
    es1 <- es1 + (d - y)^2

    v1 <- W21 %*% x
    y1 <- Sigmoid(v1)
    v <- W22 %*% y1
    y <- Sigmoid(v)
    es2 <- es2 + (d - y)^2
  }
  E1[epoch] <- es1 / N
  E2[epoch] <- es2 / N
}

plot(x=c(), y=c(), xlab="Epoch", ylab="Average of Training error", xlim=c(0, 1000), ylim=c(0, 1))
lines(E1, col="red")
lines(E2, col="blue", lty="dotted")
legend(x=0, y=1, legend=c("Cross Entropy", "Sum of Squared Error"), col=c("red", "blue"), lty=c("solid", "dotted"))
