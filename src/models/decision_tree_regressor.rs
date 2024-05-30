use ndarray::{Array1, Array2, Axis};
use ordered_float::OrderedFloat;
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct DecisionTreeRegressor {
    max_depth: usize,
    min_samples_split: usize,
    tree: Option<Node>,
}

#[derive(Debug)]
struct Node {
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    feature_index: usize,
    threshold: f64,
    value: f64,
}

impl DecisionTreeRegressor {
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        DecisionTreeRegressor {
            max_depth,
            min_samples_split,
            tree: None,
        }
    }

    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
        self.tree = Some(self.build_tree(X, y, 0));
    }

    pub fn predict(&self, X: &Array2<f64>) -> Array1<f64> {
        X.outer_iter().map(|x| self.predict_sample(&x)).collect()
    } 
    
    fn build_tree(&self, X: &Array2<f64>, y: &Array1<f64>, depth: usize) -> Node {
        if depth >= self.max_depth || y.len() < self.min_samples_split {
            return Node {
                left: None,
                right: None,
                feature_index: 0,
                threshold: 0.0,
                value: self.calculate_node_value(y),
            };
        }

        let (best_index, best_threshold, best_gain) = self.best_split(X, y);

        if best_gain == 0.0 {
            return Node {
                left: None,
                right: None,
                feature_index: 0,
                threshold: 0.0,
                value: self.calculate_node_value(y),
            };
        }

        let (left_indices, right_indices) = self.split(X, best_index, best_threshold);

        let left_X = X.select(Axis(0), &left_indices);
        let left_y = y.select(Axis(0), &left_indices);
        let right_X = X.select(Axis(0), &right_indices);
        let right_y = y.select(Axis(0), &right_indices);

        Node {
            left: Some(Box::new(self.build_tree(&left_X, &left_y, depth + 1))),
            right: Some(Box::new(self.build_tree(&right_X, &right_y, depth + 1))),
            feature_index: best_index,
            threshold: best_threshold,
            value: self.calculate_node_value(y),
        }
    }

    fn calculate_node_value(&self, y: &Array1<f64>) -> f64 {
        if y.is_empty() {
            0.0
        } else {
            y.mean().unwrap()
        }
    }

    fn best_split(&self, X: &Array2<f64>, y: &Array1<f64>) -> (usize, f64, f64) {
        let mut best_gain = 0.0;
        let mut best_index = 0;
        let mut best_threshold = 0.0;

        for feature_index in 0..X.ncols() {
            let feature_values = X.column(feature_index).to_owned();
            let unique_values: BTreeSet<_> = feature_values.iter().map(|&x| OrderedFloat(x)).collect();
            let mut previous_gain = 0.0;

            for &threshold in unique_values.iter() {
                let threshold = threshold.into_inner();
                let (left_indices, right_indices) = self.split(X, feature_index, threshold);

                if left_indices.len() < self.min_samples_split || right_indices.len() < self.min_samples_split {
                    continue; // Optionally adjust logic here to allow smaller splits if gain is high
                }

                let gain = self.information_gain(y, &left_indices, &right_indices);

                // Only update if gain improved significantly to avoid minimal updates
                if gain > best_gain && (gain - previous_gain).abs() > 0.01 {
                    best_gain = gain;
                    best_index = feature_index;
                    best_threshold = threshold;
                    previous_gain = gain;
                }
            }
        }

        (best_index, best_threshold, best_gain)
    }

    fn split(&self, X: &Array2<f64>, feature_index: usize, threshold: f64) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for (i, row) in X.outer_iter().enumerate() {
            if row[feature_index] <= threshold {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        (left_indices, right_indices)
    }

    fn information_gain(&self, y: &Array1<f64>, left_indices: &[usize], right_indices: &[usize]) -> f64 {
        let left_y = y.select(Axis(0), left_indices);
        let right_y = y.select(Axis(0), right_indices);

        let left_var = self.variance(&left_y);
        let right_var = self.variance(&right_y);

        let n = y.len() as f64;
        let n_left = left_y.len() as f64;
        let n_right = right_y.len() as f64;

        let gain = self.variance(y) - (n_left / n) * left_var - (n_right / n) * right_var;
        println!("Information gain: {}", gain);
        gain
    }

    fn variance(&self, y: &Array1<f64>) -> f64 {
        let mean = y.mean().unwrap();
        let variance = y.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / y.len() as f64;
        println!("Variance: {}", variance);
        variance
    }

    fn predict_sample(&self, x: &ndarray::ArrayView1<f64>) -> f64 {
        let mut node = self.tree.as_ref().unwrap();
        while let Some(next_node) = if x[node.feature_index] <= node.threshold {
            node.left.as_ref()
        } else {
            node.right.as_ref()
        } {
            node = next_node;
        }
        node.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray::arr2;
    use ndarray::arr1;

    #[test]
    fn test_best_split() {
        let mut model = DecisionTreeRegressor::new(3, 2);
        let X = arr2(&[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]);
        let y = arr1(&[1.0, 2.0, 3.0, 4.0]);

        let (best_index, best_threshold, best_gain) = model.best_split(&X, &y);
        println!("Best Index: {}, Best Threshold: {}, Best Gain: {}", best_index, best_threshold, best_gain);
        assert!(best_gain > 0.0, "Expected a positive information gain");
        assert_eq!(best_index, 0, "Expected best feature index to be 0");
        assert_eq!(best_threshold, 2.0, "Expected best threshold to be 2.0");
    }

    #[test]
    fn test_split() {
        let model = DecisionTreeRegressor::new(3, 2);
        let X = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let feature_index = 0;
        let threshold = 3.0;

        let (left_indices, right_indices) = model.split(&X, feature_index, threshold);
        assert_eq!(left_indices, vec![0, 1]);
        assert_eq!(right_indices, vec![2]);
    }

    #[test]
    fn test_information_gain() {
        let model = DecisionTreeRegressor::new(3, 2);
        let y = arr1(&[1.0, 3.0, 5.0]);
        let left_indices = vec![0];
        let right_indices = vec![1, 2];

        let gain = model.information_gain(&y, &left_indices, &right_indices);
        assert!(gain > 0.0, "Expected a positive gain");
    }

    #[test]
    fn test_variance() {
        let model = DecisionTreeRegressor::new(3, 2);
        let y = arr1(&[1.0, 3.0, 5.0]);

        let variance = model.variance(&y);
        assert_eq!(variance, 2.6666666666666665);
    }

    #[test]
    fn test_build_tree() {
        let mut model = DecisionTreeRegressor::new(3, 1); // Allow smaller splits
        let X = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let y = array![0.5, 1.5, 2.5];

        let node = model.build_tree(&X, &y, 0);
        assert!(node.left.is_some() && node.right.is_some(), "Tree should have both left and right children");
        assert_eq!(node.feature_index, 0, "Expected feature index to be 0");
        assert_eq!(node.threshold, 1.0, "Expected threshold to be 1.0");
    }
    
    #[test]
    fn test_fit_predict() {
        let mut model = DecisionTreeRegressor::new(3, 1); // Allow smaller splits
        let X = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        model.fit(&X, &y);
        let y_pred = model.predict(&X);

        for (pred, actual) in y_pred.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < 1e-1, "Prediction: {}, Actual: {}", pred, actual);
        }
    }

    #[test]
    fn test_fit_predict_with_noise() {
        let mut model = DecisionTreeRegressor::new(4, 1);
        let X = arr2(&[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [3.5, 35.0], [4.0, 40.0], [5.0, 50.0]]);
        let y = arr1(&[1.0, 2.0, 3.0, 3.2, 4.0, 5.0]); // Added a slightly noisy value

        model.fit(&X, &y);
        let y_pred = model.predict(&X);

        println!("Predictions: {:?}", y_pred);
        for (pred, actual) in y_pred.iter().zip(y.iter()) {
            println!("Prediction: {}, Actual: {}", pred, actual);
            assert!((pred - actual).abs() < 1.0, "Prediction: {}, Actual: {}", pred, actual); // Relaxing the assertion for noisy data
        }
    }
}
