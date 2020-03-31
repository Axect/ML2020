library(matlib)

# phi_j
phi <- function(j, s, x) {
    if (j == 0) {
        return(1)
    }
    return(exp(- (x - j)^2 / (2*s^2)))
}

# phi_vec : 101 x 1
phi_vec <- function(s, x) {
    v <- rep(0, 101)
    for (i in 1:101) {
        v[i] <- phi(i - 1, s, x)
    }
    return(matrix(v, byrow = F, nrow = length(v)))
}

# design matrix : 100 x 101
phi_mat <- function(s) {
    m <- matrix(0, 100, 101)
    for (i in 1:100) {
        for (j in 1:101) {
            m[i, j] <- phi(j - 1, s, i)
        }
    }
    return(m)
}

# w : 101 x 1
w_mle <- function(s, t) {
    pm <- phi_mat(s)
    t_mat <- matrix(t, byrow = F, nrow = length(t))
    gpm <- matlib::Ginv(pm)
    return(gpm %*% t_mat)
}

# y
y <- function(s, w, x) {
    pv <- phi_vec(s, x)
    return((t(w) %*% pv)[1, 1])
}

f <- function(x) {
    return(sin(x / 10) + (x / 50)^2)
}

# Generate samples
eps <- 0.2 * rnorm(100)
x_sample <- seq(1, 100, 1)
y_sample <- f(x_sample) + eps

# Calculate w
s <- 1
x_reg <- seq(1, 100, 0.1)
w_ml <- w_mle(s, y_sample)
print(y(s, w_ml, 1))
y_reg <- seq(1, 100, 0.1)
for (i in 1:length(x_reg)) {
    y_reg[i] <- y(s, w_ml, x_reg[i])
}



plot(x_sample, y_sample, xlim=range(c(x_sample, x_reg)), ylim=range(c(y_sample, y_reg)), type="p")
lines(x_reg, y_reg)
