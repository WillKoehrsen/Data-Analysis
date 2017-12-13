round_quarter <- function(x) {
  x <- round(x, 3)
  base <- round(x)
  decimal <- abs(as.numeric(strsplit(as.character(x), "\\.")[[1]][2]) / 1000)
  quarters <- c(0.0, 0.25, 0.5, 0.75, 1.0)
  new_decimal <- quarters[which.min(abs(quarters - decimal))]
  if (x < 0) {
    if (decimal > 0.5) {
      result <- ceiling(x) - new_decimal
    } else {
      result <- base - new_decimal
    }
  } else {
    if (decimal > 0.5) {
      result <- floor(x) + new_decimal
    } else {
      result <- base + new_decimal 
    }
  }
  result
}