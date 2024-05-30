use ndarray::{concatenate, Array1, Array2, Axis};

pub struct PolynomialFeatures {
    degree: usize,
}

impl PolynomialFeatures {
    pub fn new(degree: usize) -> Self {
        PolynomialFeatures { degree }
    }

    pub fn fit_transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut x_poly = x.clone();
        for d in 2..=self.degree {
            let x_d = x.mapv(|x| x.powi(d as i32));
            x_poly = concatenate![Axis(1), x_poly, x_d];
        }
        x_poly
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
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];

        let mut scaler = StandardScaler::new();
        scaler.fit(&x);

        assert!(scaler.mean.is_some());
        assert!(scaler.std_dev.is_some());
    }

    #[test]
    fn test_transform() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];

        let mut scaler = StandardScaler::new();
        scaler.fit(&x);
        let x_scaled = scaler.transform(&x);

        let expected_scaled = array![
            [-1.224744871391589, -1.224744871391589, -1.224744871391589],
            [0.0, 0.0, 0.0],
            [1.224744871391589, 1.224744871391589, 1.224744871391589]
        ];

        for (scaled_row, expected_row) in x_scaled.outer_iter().zip(expected_scaled.outer_iter()) {
            for (scaled, expected) in scaled_row.iter().zip(expected_row.iter()) {
                assert!((scaled - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_fit_transform() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];

        let mut scaler = StandardScaler::new();
        let x_scaled = scaler.fit_transform(&x);

        let expected_scaled = array![
            [-1.224744871391589, -1.224744871391589, -1.224744871391589],
            [0.0, 0.0, 0.0],
            [1.224744871391589, 1.224744871391589, 1.224744871391589]
        ];

        for (scaled_row, expected_row) in x_scaled.outer_iter().zip(expected_scaled.outer_iter()) {
            for (scaled, expected) in scaled_row.iter().zip(expected_row.iter()) {
                assert!((scaled - expected).abs() < 1e-6);
            }
        }
    }
}