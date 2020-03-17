extern crate peroxide;
use peroxide::*;
use std::f64::consts::PI;

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

    let x = rnorm!(100, 2, 0.1);
    let y = rnorm!(100, 4, 0.3);
    x.mean().print();
    y.mean().print();
    let mut dg = DataFrame::with_header(vec!["x", "y"]);
    dg["x"] = x.clone();
    dg["y"] = y.clone();
    let m = hstack!(x, y);
    let df1 = equi_mahal(&m, 1f64);
    let df2 = equi_mahal(&m, 2f64);
    df1.print();
    df1.write_nc("mahal1.nc").expect("Can't write nc");
    df2.write_nc("mahal2.nc").expect("Can't write nc2");
    dg.write_nc("data.nc").expect("Can't write data");
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

fn equi_mahal(m: &Matrix, d: f64) -> DataFrame {
    let mu = m.mean();
    let sigma = m.cov();
    let eig = eigen(&sigma, Jacobi);
    let (ev, mut em) = eig.extract();
    ev.print();
    em.print();
    em.col(0).dot(&em.col(1)).print();
    em.col(0).dot(&em.col(0)).print();
    let lam1 = ev[1];
    let lam2 = ev[0];
    let th = seq(0f64, 2f64 * PI, 0.01);
    let y1 = th.fmap(|t| t.cos() * lam1.sqrt() * d);
    let y2 = th.fmap(|t| t.sin() * lam2.sqrt() * d);
    let y = vstack!(y1, y2);
    unsafe {
        em.swap(0, 1, Col);
    }
    let u = em.t();
    let mut x = u.inv().unwrap() * y;
    x.col_mut_map(|t| t.add(&mu));
    let x = x.t();
    let mut df = DataFrame::with_header(vec!["x", "y"]);
    df["x"] = x.col(0);
    df["y"] = x.col(1);
    df
}
