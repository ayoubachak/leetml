use ndarray::{s, Array1, Array2, Axis, concatenate};
use serde::{Deserialize, Serialize};
use crate::preprocessing::PolynomialFeatures;

use super::utils::{invert_matrix, EPSILON};


#[derive(Serialize, Deserialize)]
pub struct LinearRegression {
    coefficients: Option<Array1<f64>>,
    intercept: f64,
    degree: usize,
}

impl LinearRegression {
    pub fn new(degree: usize) -> Self {
        LinearRegression {
            coefficients: None,
            intercept: 0.0,
            degree,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let poly = PolynomialFeatures::new(self.degree);
        let x_poly = poly.fit_transform(&x);

        let ones = Array2::ones((x_poly.nrows(), 1));
        let x_b = concatenate![Axis(1), ones.view(), x_poly.view()];

        let xt_x = x_b.t().dot(&x_b);
        let xt_y = x_b.t().dot(y);

        let xt_x_inv = invert_matrix(xt_x.clone(), EPSILON).expect("Failed to invert matrix");

        let beta = xt_x_inv.dot(&xt_y);

        self.intercept = beta[0];
        self.coefficients = Some(beta.slice(s![1..]).to_owned());
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let poly = PolynomialFeatures::new(self.degree);
        let x_poly = poly.fit_transform(&x);

        let mut y_pred = Array1::from_elem(x_poly.nrows(), self.intercept);
        if let Some(ref coef) = self.coefficients {
            y_pred += &x_poly.dot(coef);
        }
        y_pred
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = serde_json::to_string(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let model: Self = serde_json::from_str(&data)?;
        Ok(model)
    }

}



#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_fit() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut model = LinearRegression::new(1);
        model.fit(&x, &y);

        assert!(model.coefficients.is_some());
        assert!((model.intercept - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_predict() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut model = LinearRegression::new(1);
        model.fit(&x, &y);

        let y_pred = model.predict(&x);
        for (pred, actual) in y_pred.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < EPSILON);
        }
    }

    #[test]
    fn test_save_and_load() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut model = LinearRegression::new(1);
        model.fit(&x, &y);
        model.save("test_model.json").expect("Failed to save model");

        let loaded_model = LinearRegression::load("test_model.json").expect("Failed to load model");
        let y_pred = loaded_model.predict(&x);

        for (pred, actual) in y_pred.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < EPSILON);
        }

        std::fs::remove_file("test_model.json").expect("Failed to delete test model file");
    }

    #[test]
    fn test_polynomial_features() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0]
        ];
        let y = array![1.0, 8.0, 27.0, 64.0, 125.0]; // y = x^3

        let mut model = LinearRegression::new(3);
        model.fit(&x, &y);

        let y_pred = model.predict(&x);
        for (pred, actual) in y_pred.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < EPSILON);
        }
    }
}