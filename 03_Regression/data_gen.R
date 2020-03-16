x <- seq(0, 10, 0.1)
y <- 2 * x  - 1 + rnorm(length(x))

df <- data.frame(X = x, Y = y)

write.table(df, file="data.csv", sep = ",", row.names=FALSE)
