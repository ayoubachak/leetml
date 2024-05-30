use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand::{self, Rng};
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;

use super::decision_tree_regressor::DecisionTreeRegressor;

#[derive(Debug)]
pub struct RandomForestRegressor {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    trees: Vec<DecisionTreeRegressor>,
    random_seed: Option<u64>,
}

impl RandomForestRegressor {
    pub fn new(n_estimators: usize, max_depth: Option<usize>, min_samples_split: usize, random_seed: Option<u64>) -> Self {
        RandomForestRegressor {
            n_estimators,
            max_depth,
            min_samples_split,
            trees: Vec::new(),
            random_seed,
        }
    }

    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
        let mut rng: StdRng = match self.random_seed {
            Some(seed) => SeedableRng::seed_from_u64(seed),
            None => SeedableRng::from_entropy(),
        };

        for i in 0..self.n_estimators {
            let bootstrap_sample: Vec<usize> = (0..X.nrows()).map(|_| rng.gen_range(0..X.nrows())).collect();

            let X_bootstrap = X.select(Axis(0), &bootstrap_sample);
            let y_bootstrap = y.select(Axis(0), &bootstrap_sample);

            // Debugging prints
            println!("Bootstrap sample {}: {:?}", i, bootstrap_sample);
            println!("X_bootstrap {}: {:?}", i, X_bootstrap);
            println!("y_bootstrap {}: {:?}", i, y_bootstrap);

            let mut tree = DecisionTreeRegressor::new(self.max_depth.unwrap_or(usize::MAX), self.min_samples_split);
            tree.fit(&X_bootstrap, &y_bootstrap);

            self.trees.push(tree);
        }
    }

    pub fn predict(&self, X: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Array2::<f64>::zeros((self.n_estimators, X.nrows()));
        for (i, tree) in self.trees.iter().enumerate() {
            let tree_predictions = tree.predict(X);
            predictions.row_mut(i).assign(&tree_predictions);

            // Debugging prints
            println!("Tree {} predictions: {:?}", i, tree_predictions);
        }

        let mean_predictions = predictions.mean_axis(Axis(0)).unwrap();

        // Debugging print
        println!("Mean predictions: {:?}", mean_predictions);

        mean_predictions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr1};

    #[test]
    fn test_fit_predict() {
        let X = arr2(&[
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ]);
        let y = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut model = RandomForestRegressor::new(10, Some(3), 2, Some(42));
        model.fit(&X, &y);
        let y_pred = model.predict(&X);

        println!("Predictions: {:?}", y_pred);
        for (pred, actual) in y_pred.iter().zip(y.iter()) {
            assert!((pred - actual).abs() <= 1.1, "Prediction: {}, Actual: {}", pred, actual);
        }
    }

    #[test]
    fn test_fit_predict_with_noise() {
        let X = arr2(&[
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ]);
        let y = arr1(&[1.1, 1.9, 3.0, 4.1, 4.9]); // Added slight noise

        let mut model = RandomForestRegressor::new(10, Some(3), 2, Some(123));
        model.fit(&X, &y);
        let y_pred = model.predict(&X);

        println!("Predictions with noise: {:?}", y_pred);
        for (pred, actual) in y_pred.iter().zip(y.iter()) {
            assert!((pred - actual).abs() <= 1.1, "Prediction: {}, Actual: {}", pred, actual);
        }
    }

    #[test]
    fn test_fit_predict_different_data() {
        let X = arr2(&[
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0],
        ]);
        let y = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut model = RandomForestRegressor::new(10, Some(3), 2, Some(456));
        model.fit(&X, &y);
        let y_pred = model.predict(&X);

        println!("Predictions with different data: {:?}", y_pred);
        for (pred, actual) in y_pred.iter().zip(y.iter()) {
            assert!((pred - actual).abs() <= 1.1, "Prediction: {}, Actual: {}", pred, actual);
        }
    }
}
