rm(list=ls())

source("./loadMNISTImages.R")
source("./loadMNISTLabels.R")
source("./rng.R")
source("./MnistConv.R")
source("./Conv.R")
source("./ReLU.R")
source("./Pool.R")
source("./Softmax.R")

Images <- loadMNISTImages("./MNIST/mnist_test.csv")
Labels <- loadMNISTLabels("./MNIST/mnist_test.csv")
Labels[Labels == 0] <- 10 # 0 --> 10

rng(1)

# Learning
#
W1 <- array(rnorm(9 * 9 * 20), c(9, 9, 20)) * 0.01
W5 <- array(runif(100 * 2000, min=-1, max=1), c(100, 2000)) * sqrt(6) / sqrt(360 + 2000)
Wo <- array(runif(10 * 100, min=-1, max=1), c(10, 100)) * sqrt(6) / sqrt(10 + 100)

X <- Images[, , 1:8000]
D <- Labels[1:8000]

for (epoch in 1:3) {
  print(paste("epoch ", epoch, sep=""))
  Ws <- MnistConv(W1, W5, Wo, X, D)
  W1 <- Ws$W1
  W5 <- Ws$W5
  Wo <- Ws$Wo
}

save.image()

# Test
#
X <- Images[, , 8001:10000]
D <- Labels[8001:10000]

acc <- 0
N <- length(D)
for (k in 1:N) {
  x <- X[, , k]                               # Input,       28x28

  y1 <- Conv(x, W1)                           # Convolution, 20x20x20
  y2 <- ReLU(y1)                              #
  y3 <- Pool(y2)                              # Pool,        10x10x20
  y4 <- as.vector(y3)                         #              2000
  v5 <- W5 %*% y4                             # ReLU,        100
  y5 <- ReLU(v5)                              #
  v <- Wo %*% y5                              # Softmax,     10
  y <- Softmax(v)                             #

  i <- which.max(y)
  if (i == D[k]) {
    acc <- acc + 1
  }
}

acc <- acc / N
print(paste("Accuracy is ", acc, sep=""))
