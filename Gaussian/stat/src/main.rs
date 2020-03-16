extern crate peroxide;
use peroxide::*;

fn main() {
    let df = DataFrame::read_csv("data.csv", ',').expect("Can't read csv");
    df.print();

    let m = df.to_matrix();
    println!();
    cov_mat(&m).print();
    println!();
    m.cov().print();
    println!("");
    mahalanobis(&c!(66, 640, 44), &m).print();
}

fn cov_mat(x: &Matrix) -> Matrix {
    let mut m = zeros(x.col, x.col);
    for i in 0 .. x.col {
        for j in 0 .. x.col {
            m[(i,j)] = cov2(&x.col(i), &x.col(j));
        }
    }
    return m
}

fn cov2(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    // Calculate mean
    let mut mx = 0f64;
    let mut my = 0f64;
    for i in 0 .. x.len() {
        mx += x[i];
        my += y[i];
    }
    mx /= x.len() as f64;
    my /= y.len() as f64;

    // Calculate cov
    let mut s = 0f64;
    for i in 0 .. x.len() {
        s += (x[i] - mx) * (y[i] - my)
    }
    s /= x.len() as f64 - 1f64;
    return s
}

fn mahalanobis(x: &Vec<f64>, m: &Matrix) -> f64 {
    let mu = m.mean();
    let sigma = m.cov();
    let x_mu = x.sub(&mu).to_matrix();
    (&x_mu.t() * &sigma.inv().unwrap() * x_mu)[(0, 0)].sqrt()
}
