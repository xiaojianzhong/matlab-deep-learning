loadMNISTLabels <- function(filename) {
  # loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
  # the labels for the MNIST images

  csv <- read.csv(filename, header=FALSE)
  labels <- matrix(csv[, 1], nrow=nrow(csv), ncol=1)
  return(labels)
}
