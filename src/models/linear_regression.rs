use ndarray::{s, Array1, Array2, Axis, concatenate};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use crate::preprocessing::PolynomialFeatures;

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

    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
        let poly = PolynomialFeatures::new(self.degree);
        let X_poly = poly.fit_transform(&X);

        let ones = Array2::ones((X_poly.nrows(), 1));
        let X_b = concatenate![Axis(1), ones.view(), X_poly.view()];

        let XtX = X_b.t().dot(&X_b);
        let XtY = X_b.t().dot(y);

        let XtX_inv = invert_matrix(XtX.clone()).expect("Failed to invert matrix");

        let beta = XtX_inv.dot(&XtY);

        self.intercept = beta[0];
        self.coefficients = Some(beta.slice(s![1..]).to_owned());
    }

    pub fn predict(&self, X: &Array2<f64>) -> Array1<f64> {
        let poly = PolynomialFeatures::new(self.degree);
        let X_poly = poly.fit_transform(&X);

        let mut y_pred = Array1::from_elem(X_poly.nrows(), self.intercept);
        if let Some(ref coef) = self.coefficients {
            y_pred += &X_poly.dot(coef);
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

// Helper function to swap rows of a matrix
fn swap_rows(matrix: &mut Array2<f64>, row1: usize, row2: usize) {
    if row1 != row2 {
        for col in 0..matrix.ncols() {
            matrix.swap((row1, col), (row2, col));
        }
    }
}

// Function to perform Gaussian elimination and back substitution to invert a matrix
fn invert_matrix(mut matrix: Array2<f64>) -> Option<Array2<f64>> {
    let n = matrix.nrows();
    let mut identity = Array2::eye(n);
    let epsilon = 1e-10; // Small value to add to the diagonal elements

    for i in 0..n {
        matrix[(i, i)] += epsilon;
    }

    for i in 0..n {
        // Debug: Print the matrix and identity matrix at the start of each iteration

        // Find the pivot row
        let mut pivot = i;
        for j in i + 1..n {
            if matrix[(j, i)].abs() > matrix[(pivot, i)].abs() {
                pivot = j;
            }
        }

        if matrix[(pivot, i)].abs() < 1e-10 {
            return None;
        }

        // Swap rows in both matrices
        swap_rows(&mut matrix, i, pivot);
        swap_rows(&mut identity, i, pivot);

        // Normalize the pivot row
        let pivot_value = matrix[(i, i)];
        for j in 0..n {
            matrix[(i, j)] /= pivot_value;
            identity[(i, j)] /= pivot_value;
        }

        // Debug: Print the matrix and identity matrix after normalization

        // Eliminate the current column
        for j in 0..n {
            if j != i {
                let factor = matrix[(j, i)];
                for k in 0..n {
                    matrix[(j, k)] -= factor * matrix[(i, k)];
                    identity[(j, k)] -= factor * identity[(i, k)];
                }
            }
        }

        // Debug: Print the matrix and identity matrix after elimination
    }

    Some(identity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_fit() {
        let X = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut model = LinearRegression::new(1);
        model.fit(&X, &y);

        assert!(model.coefficients.is_some());
        assert!((model.intercept - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_predict() {
        let X = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut model = LinearRegression::new(1);
        model.fit(&X, &y);

        let y_pred = model.predict(&X);
        for (pred, actual) in y_pred.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < EPSILON);
        }
    }

    #[test]
    fn test_save_and_load() {
        let X = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut model = LinearRegression::new(1);
        model.fit(&X, &y);
        model.save("test_model.json").expect("Failed to save model");

        let loaded_model = LinearRegression::load("test_model.json").expect("Failed to load model");
        let y_pred = loaded_model.predict(&X);

        for (pred, actual) in y_pred.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < EPSILON);
        }

        std::fs::remove_file("test_model.json").expect("Failed to delete test model file");
    }

    #[test]
    fn test_polynomial_features() {
        let X = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0]
        ];
        let y = array![1.0, 8.0, 27.0, 64.0, 125.0]; // y = x^3

        let mut model = LinearRegression::new(3);
        model.fit(&X, &y);

        let y_pred = model.predict(&X);
        for (pred, actual) in y_pred.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < EPSILON);
        }
    }
}