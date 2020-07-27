extern crate rand;
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

fn main() {
    let (h, t) = coin_flip(10);
    println!("{}, {}", h, t);
}

fn coin_flip(n: usize) -> (usize, usize) {
	let mut h = 0usize;
    let mut t = 0usize;
    for _i in 0 .. n {
   		match rand::random() {
        	State::H => { h += 1; },
            State::T => { t += 1; }
        }
    }
    (h, t)
}

#[derive(Debug, Copy, Clone)]
enum State {
    H,
    T
}

impl Distribution<State> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> State {
        match rng.gen_range(0, 2) {
            0 => State::H,
            1 => State::T,
            _ => unreachable!()
        }
    }
}
