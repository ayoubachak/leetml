use ndarray::{Array1, Array2, Axis, s};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::{rand, RandomExt};

#[derive(Debug)]
pub struct LogisticRegression {
    coefficients: Option<Array1<f64>>,
    intercept: f64,
    learning_rate: f64,
    iterations: usize,
}

impl LogisticRegression {
    pub fn new(learning_rate: f64, iterations: usize) -> Self {
        LogisticRegression {
            coefficients: None,
            intercept: 0.0,
            learning_rate,
            iterations,
        }
    }

    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
        let mut rng = rand::thread_rng();
        let n_features = X.ncols();
        let mut coefficients = Array1::random_using(n_features, Uniform::new(0., 1.), &mut rng);
        let mut intercept = 0.0;

        for _ in 0..self.iterations {
            let linear_model = X.dot(&coefficients) + intercept;
            let predictions = linear_model.mapv(sigmoid);

            let errors = y - &predictions;

            coefficients = coefficients + &(X.t().dot(&errors) * self.learning_rate);
            intercept += errors.sum() * self.learning_rate;
        }

        self.coefficients = Some(coefficients);
        self.intercept = intercept;
    }

    pub fn predict(&self, X: &Array2<f64>) -> Array1<f64> {
        let linear_model = X.dot(self.coefficients.as_ref().unwrap()) + self.intercept;
        linear_model.mapv(sigmoid).mapv(|p| if p >= 0.5 { 1.0 } else { 0.0 })
    }

    pub fn predict_proba(&self, X: &Array2<f64>) -> Array1<f64> {
        let linear_model = X.dot(self.coefficients.as_ref().unwrap()) + self.intercept;
        linear_model.mapv(sigmoid)
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray::arr2;
    use ndarray::arr1;

    #[test]
    fn test_logistic_regression_fit() {
        let X = arr2(&[
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 1.0],
            [5.0, 1.0]
        ]);
        let y = arr1(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = LogisticRegression::new(0.1, 1000);
        model.fit(&X, &y);

        let coefficients = model.coefficients.as_ref().unwrap();
        println!("Coefficients: {:?}", coefficients);
        println!("Intercept: {}", model.intercept);
        assert!(coefficients.len() > 0, "Coefficients should not be empty");
    }

    #[test]
    fn test_logistic_regression_predict() {
        let X = arr2(&[
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 1.0],
            [5.0, 1.0]
        ]);
        let y = arr1(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = LogisticRegression::new(0.1, 1000);
        model.fit(&X, &y);

        let predictions = model.predict(&X);
        println!("Predictions: {:?}", predictions);
        assert_eq!(predictions.len(), y.len(), "Predictions should have the same length as the input");

        for (pred, actual) in predictions.iter().zip(y.iter()) {
            assert_eq!(*pred, *actual, "Prediction: {}, Actual: {}", pred, actual);
        }
    }

    #[test]
    fn test_logistic_regression_predict_proba() {
        let X = arr2(&[
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 1.0],
            [5.0, 1.0]
        ]);
        let y = arr1(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = LogisticRegression::new(0.1, 1000);
        model.fit(&X, &y);

        let probabilities = model.predict_proba(&X);
        println!("Probabilities: {:?}", probabilities);
        assert_eq!(probabilities.len(), y.len(), "Probabilities should have the same length as the input");

        for proba in probabilities.iter() {
            assert!((*proba >= 0.0) && (*proba <= 1.0), "Probability should be between 0 and 1");
        }
    }
}
