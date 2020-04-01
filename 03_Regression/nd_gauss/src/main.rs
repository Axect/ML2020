#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;
use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray_rand::rand::thread_rng;
use ndarray_rand::rand_distr::*;

fn main() {
    let sample = gen_sample();
    let x = sample.row(0);
    let y = sample.row(1);
    println!("{}", x);
    println!("{}", y);
}

fn f(x: f64) -> f64 {
    (x / 10f64).sin() + (x / 50f64).powi(2)
}

fn gen_sample() -> Array2<f64> {
    let normal = Normal::new(0., 1.).unwrap();
    let e: Array1<f64> = normal.sample_iter(&mut thread_rng()).take(100).collect();
    let x: Array1<f64> = Array::linspace(1., 100., 100);
    let y: Array1<f64> = x.mapv(|t| f(t)) + e;
    let z = stack![Axis(0), x, y];
    z.into_shape((2, 100)).unwrap()
}