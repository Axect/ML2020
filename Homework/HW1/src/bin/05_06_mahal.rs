extern crate peroxide;
use peroxide::*;
use std::f64::consts::PI;

fn main() {
    let df = DataFrame::read_csv("data.csv", ',').expect("Can't read csv");
    println!("Data:");
    df.print();

    let m = df.to_matrix();
    println!("");

    // 05 Covariance matrix & Mahalanobis Distance
    println!("Covariance: ");
    m.cov().print();
    println!("");
    println!("Mahalanobis Distance:");
    mahalanobis(&c!(66, 640, 44), &m).print();
    println!("");

    // 06 Equi-Mahalanobis Distance
    let x = rnorm!(100, 2, 0.1);
    let y = rnorm!(100, 4, 0.3);
    println!("Mean of x:");
    x.mean().print();
    println!("\nMean of y:");
    y.mean().print();
    println!("");
    let mut dg = DataFrame::with_header(vec!["x", "y"]);
    dg["x"] = x.clone();
    dg["y"] = y.clone();
    let m = hstack!(x, y);
    let df1 = equi_mahal(&m, 1f64);
    let df2 = equi_mahal(&m, 2f64);
    df1.write_nc("mahal1.nc").expect("Can't write nc");
    df2.write_nc("mahal2.nc").expect("Can't write nc2");
    dg.write_nc("data.nc").expect("Can't write data");
    println!("Write plot data to nc_file");
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
    println!("Eigenvalue:");
    ev.print();
    println!("");
    println!("Eigenvector:");
    em.print();
    println!("");
    println!("Check orthogonality: ");
    println!("i*j = {}", em.col(0).dot(&em.col(1)));
    println!("i*i = {}", em.col(0).dot(&em.col(0)));
    println!("");
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
