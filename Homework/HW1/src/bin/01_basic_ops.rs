extern crate peroxide;
use peroxide::{Matrix, MATLAB, zeros, Printable};

fn main() {
    let mut a = [0f64; 5];
    let mut d = [0f64; 5];
    for (i, (x, y)) in a.iter_mut().zip(d.iter_mut()).enumerate() {
        *x += i as f64 + 1f64;
        *y += 5f64 - i as f64;
    }
    println!("Input vec: {:?}", a);
    println!("Input vec: {:?}", d);

    let b = &a[..];
    let c = &d[..];

    println!("Mean: {}", b.mean());
    println!("Var: {}", b.var());
    println!("Cov: {}", b.cov(&c));
    println!("");

    let m: Matrix = MATLAB::new("1 8; 2 6; 3 4; 4 2");
    m.print();
    m.mean().print();
    m.var().print();
    m.cov_mat().print();
}

trait Statistics {
    type Output;
    fn mean(&self) -> Self::Output;
    fn var(&self) -> Self::Output;
    fn cov(&self, rhs: &Self) -> Self::Output;
    fn cov_mat(&self) -> Matrix;
}

impl<'a> Statistics for &'a [f64] {
    type Output = f64;
    
    fn mean(&self) -> Self::Output {
        let mut s = 0f64;
        for elem in self.iter() {
            s += elem;
        }
        s / self.len() as f64
    }

    fn var(&self) -> Self::Output {
        let mut s1 = 0f64;
        let mut s2 = 0f64;
        for elem in self.iter() {
            s1 += elem.powi(2);
            s2 += elem;
        }
        let l_f64 = self.len() as f64;
        s1 /= l_f64 - 1f64;
        s1 - s2.powi(2) / (l_f64 * (l_f64 - 1f64))
    }

    fn cov(&self, rhs: &Self) -> Self::Output {
        let mut s = 0f64;
        let mut s_x = 0f64;
        let mut s_y = 0f64;
        for (x, y) in self.iter().zip(rhs.iter()) {
            s += x * y;
            s_x += x;
            s_y += y;
        }
        let l_f64 = self.len() as f64;
        let l_f64_1 = l_f64 - 1f64;
        s /= l_f64_1;
        s_x /= l_f64;
        s_y /= l_f64_1;
        s - s_x * s_y
    }
    
    fn cov_mat(&self) -> Matrix {
        unimplemented!()
    }
}

impl Statistics for Matrix {
    type Output = Vec<f64>;

    fn mean(&self) -> Self::Output {
        let mut v = vec![0f64; self.col];
        for (i, x) in v.iter_mut().enumerate() {
            let c = self.col(i);
            let d = c.as_slice();
            *x = d.mean();
        }
        v
    }

    fn var(&self) -> Self::Output {
        let mut v = vec![0f64; self.col];
        for (i, x) in v.iter_mut().enumerate() {
            let c = self.col(i);
            let d = c.as_slice();
            *x = d.var();
        }
        v
    }

    fn cov(&self, _rhs: &Self) -> Self::Output {
        unimplemented!()
    }

    fn cov_mat(&self) -> Matrix {
        let mut m = zeros(self.col, self.col);
        for i in 0 .. m.col {
            let c_i = self.col(i);
            let c_is = c_i.as_slice();
            for j in 0 .. m.col {
                let c_j = self.col(j);
                let c_js = c_j.as_slice();
                m[(i, j)] = c_is.cov(&c_js);
            }
        }
        m
    }
}
