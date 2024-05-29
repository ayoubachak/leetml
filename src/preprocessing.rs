use ndarray::{concatenate, Array1, Array2, Axis};

pub struct PolynomialFeatures {
    degree: usize,
}

impl PolynomialFeatures {
    pub fn new(degree: usize) -> Self {
        PolynomialFeatures { degree }
    }

    pub fn fit_transform(&self, X: &Array2<f64>) -> Array2<f64> {
        let mut X_poly = X.clone();
        for d in 2..=self.degree {
            let X_d = X.mapv(|x| x.powi(d as i32));
            X_poly = concatenate![Axis(1), X_poly, X_d];
        }
        X_poly
    }
}

pub struct StandardScaler {
    mean: Option<Array1<f64>>,
    std_dev: Option<Array1<f64>>,
}

impl StandardScaler {
    pub fn new() -> Self {
        StandardScaler {
            mean: None,
            std_dev: None,
        }
    }

    pub fn fit(&mut self, data: &Array2<f64>) {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let std_dev = data.std_axis(Axis(0), 0.0);
        self.mean = Some(mean);
        self.std_dev = Some(std_dev);
    }

    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean.as_ref().unwrap();
        let std_dev = self.std_dev.as_ref().unwrap();
        let mut scaled_data = data.clone();
        for mut row in scaled_data.rows_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                *val = (*val - mean[i]) / std_dev[i];
            }
        }
        scaled_data
    }

    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fit() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];

        let mut scaler = StandardScaler::new();
        scaler.fit(&X);

        assert!(scaler.mean.is_some());
        assert!(scaler.std_dev.is_some());
    }

    #[test]
    fn test_transform() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];

        let mut scaler = StandardScaler::new();
        scaler.fit(&X);
        let X_scaled = scaler.transform(&X);

        let expected_scaled = array![
            [-1.224744871391589, -1.224744871391589, -1.224744871391589],
            [0.0, 0.0, 0.0],
            [1.224744871391589, 1.224744871391589, 1.224744871391589]
        ];

        for (scaled_row, expected_row) in X_scaled.outer_iter().zip(expected_scaled.outer_iter()) {
            for (scaled, expected) in scaled_row.iter().zip(expected_row.iter()) {
                assert!((scaled - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_fit_transform() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];

        let mut scaler = StandardScaler::new();
        let X_scaled = scaler.fit_transform(&X);

        let expected_scaled = array![
            [-1.224744871391589, -1.224744871391589, -1.224744871391589],
            [0.0, 0.0, 0.0],
            [1.224744871391589, 1.224744871391589, 1.224744871391589]
        ];

        for (scaled_row, expected_row) in X_scaled.outer_iter().zip(expected_scaled.outer_iter()) {
            for (scaled, expected) in scaled_row.iter().zip(expected_row.iter()) {
                assert!((scaled - expected).abs() < 1e-6);
            }
        }
    }
}