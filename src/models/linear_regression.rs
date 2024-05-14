use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::solve::Inverse;

pub struct LinearRegression {
    coefficients: Option<Array1<f64>>,
    intercept: f64,
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression {
            coefficients: None,
            intercept: 0.0,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        // Add a column of ones to the input matrix for the intercept term
        let ones = Array2::ones((x.nrows(), 1));
        let x_b = ndarray::concatenate![Axis(1), ones, x.view().into_shape((x.nrows(), x.ncols())).unwrap()];
    
        // Compute the pseudo-inverse of X_b
        let xt_x = x_b.t().dot(&x_b);
        let xt_y = x_b.t().dot(y);
    
        let beta = xt_x.inv().expect("Failed to invert matrix").dot(&xt_y);
    
        self.intercept = beta[0];
        self.coefficients = Some(beta.slice(s![1..]).to_owned());
    }
    
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut y_pred = Array1::from_elem(x.nrows(), self.intercept);
        if let Some(ref coef) = self.coefficients {
            y_pred += &x.dot(coef);
        }
        y_pred
    }
}
