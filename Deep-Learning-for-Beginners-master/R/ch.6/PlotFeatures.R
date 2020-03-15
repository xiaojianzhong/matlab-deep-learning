rm(list=ls())

library(plot.matrix)
source("./Conv.R")
source("./ReLU.R")
source("./Pool.R")
source("./Softmax.R")

load("./.RData")

k <- 2
x <- X[, , k]                               # Input,       28x28
y1 <- Conv(x, W1)                           # Convolution, 20x20x20
y2 <- ReLU(y1)                              #
y3 <- Pool(y2)                              # Pool,        10x10x20
y4 <- as.vector(y3)                         #              2000
v5 <- W5 %*% y4                             # ReLU,        100
y5 <- ReLU(v5)                              #
v <- Wo %*% y5                              # Softmax,     10
y <- Softmax(v)                             #

plot(x)

convFilters <- rbind(
  cbind(W1[, , 1], W1[, , 2], W1[, , 3], W1[, , 4], W1[, , 5]),
  cbind(W1[, , 6], W1[, , 7], W1[, , 8], W1[, , 9], W1[, , 10]),
  cbind(W1[, , 11], W1[, , 12], W1[, , 13], W1[, , 14], W1[, , 15]),
  cbind(W1[, , 16], W1[, , 17], W1[, , 18], W1[, , 19], W1[, , 20])
)
plot(convFilters)

fList <- rbind(
  cbind(y1[, , 1], y1[, , 2], y1[, , 3], y1[, , 4], y1[, , 5]),
  cbind(y1[, , 6], y1[, , 7], y1[, , 8], y1[, , 9], y1[, , 10]),
  cbind(y1[, , 11], y1[, , 12], y1[, , 13], y1[, , 14], y1[, , 15]),
  cbind(y1[, , 16], y1[, , 17], y1[, , 18], y1[, , 19], y1[, , 20])
)
plot(fList)

fList <- rbind(
  cbind(y2[, , 1], y2[, , 2], y2[, , 3], y2[, , 4], y2[, , 5]),
  cbind(y2[, , 6], y2[, , 7], y2[, , 8], y2[, , 9], y2[, , 10]),
  cbind(y2[, , 11], y2[, , 12], y2[, , 13], y2[, , 14], y2[, , 15]),
  cbind(y2[, , 16], y2[, , 17], y2[, , 18], y2[, , 19], y2[, , 20])
)
plot(fList)

fList <- rbind(
  cbind(y3[, , 1], y3[, , 2], y3[, , 3], y3[, , 4], y3[, , 5]),
  cbind(y3[, , 6], y3[, , 7], y3[, , 8], y3[, , 9], y3[, , 10]),
  cbind(y3[, , 11], y3[, , 12], y3[, , 13], y3[, , 14], y3[, , 15]),
  cbind(y3[, , 16], y3[, , 17], y3[, , 18], y3[, , 19], y3[, , 20])
)
plot(fList)
