extern crate peroxide;
use peroxide::*;

fn main() {
    let df = gen_sample(100);

    let x = &df["x"];
    let t = &df["t"];

    df.write_nc("data/data.nc").expect("Can't write data");

    let phi_mat = phi(x);
    let m = m_n(&phi_mat, t);
    
    let x_plot = seq(-1, 1, 0.1);
    let y_plot = x_plot.fmap(|x| m[(0, 0)] + m[(1, 0)] * x);

    let mut dg = DataFrame::with_header(vec!["x", "y"]);
    dg["x"] = x_plot;
    dg["y"] = y_plot;

    dg.write_nc("data/reg.nc").expect("Can't write reg");
}

fn gen_sample(n: usize) -> DataFrame {
    let unif = Uniform(-1f64, 1f64);
    let x = unif.sample(n);
    let norm = Normal(0f64, 0.2);
    let eps = norm.sample(100);
    let mut t = x.fmap(f);
    t.mut_zip_with(|x, y| x + y, &eps);

    let mut df = DataFrame::with_header(vec!["x", "t"]);
    df["x"] = x;
    df["t"] = t;
    df
}

fn f(x: f64) -> f64 {
    -0.3 + 0.5 * x
}

fn phi(x: &Vec<f64>) -> Matrix {
    let n = x.len();
    let mut ph = matrix(vec![1f64; 2*n], n, 2, Col);
    for i in 0 .. n {
        ph[(i, 1)] = x[i];
    }
    ph
}

fn s_n_inv(ph: &Matrix) -> Matrix {
    2f64 * eye(2) + 25f64 * (&ph.t() * ph)
}

fn m_n(ph: &Matrix, t: &Vec<f64>) -> Matrix {
    let s = s_n_inv(ph).inv().unwrap();
    s * ph.t() * t.fmap(|x| x * 25f64).to_matrix()
}
