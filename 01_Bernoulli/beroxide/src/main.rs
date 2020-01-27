extern crate peroxide;
use peroxide::*;
use std::env::args;

fn main() {
    // Receive Arguments
    let args: Vec<String> = args().collect();
    let n: usize = args[1].parse().unwrap();
    
    // Generate Data
    let d = gen_flip_array(n);

    // Maximum Likelihood Estimation
    let mu_ml = mle(&d);

    // Bayesian Analysis
    let prior = Beta(2f64, 2f64);
    let posterior = bayesian_update(&prior, &d);

    print!("mle: ");
    mu_ml.print();
    print!("bayesian: ");
    optimal_mu(&posterior).print();
}

fn gen_flip_array(n: usize) -> Vec<f64> {
    let b = Bernoulli(0.5);
    b.sample(n)
}

// Maximum Likelihood Estimation for Bernoulli
fn mle(d: &Vec<f64>) -> f64 {
    let l = d.len();
    let s = sum(&d);
    s / (l as f64)
}

// Bayesian update for Beta
fn bayesian_update(prior: &TPDist<f64>, d: &Vec<f64>) -> TPDist<f64> {
    let m = sum(&d);
    let l = (d.len() as f64) - m;
    match prior {
        Beta(a, b) => Beta(a + m, b + l),
        _ => unreachable!()
    }
}

fn sum(d: &Vec<f64>) -> f64 {
    let mut s = 0f64;
    for x in d {
        s += x;
    }
    s
}

fn optimal_mu(posterior: &TPDist<f64>) -> f64 {
    match posterior {
        Beta(a, b) => (a - 1f64) / (a + b - 2f64),
        _ => unreachable!()
    }
}
