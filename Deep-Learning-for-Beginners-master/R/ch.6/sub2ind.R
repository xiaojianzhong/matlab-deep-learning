sub2ind <- function(dim, r, c) {
  m <- dim[1]
  ind <- (c - 1) * m + r
  return(ind)
}
