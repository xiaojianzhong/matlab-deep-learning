loadMNISTImages <- function(filename) {
  # loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
  # the raw MNIST images

  csv <- read.csv(filename, header=FALSE)
  images <- aperm(array(t(csv[, 2:785] / 255), c(28, 28, nrow(csv))), c(2, 1, 3))
  return(images)
}
