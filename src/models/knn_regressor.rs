use ndarray::{Array1, Array2};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[derive(Debug)]
pub struct KNNRegressor {
    k: usize,
    training_data: Option<Array2<f64>>,
    training_labels: Option<Array1<f64>>,
}

#[derive(PartialEq)]
struct Neighbor {
    distance: f64,
    label: f64,
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.partial_cmp(&self.distance).unwrap()
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl KNNRegressor {
    pub fn new(k: usize) -> Self {
        KNNRegressor {
            k,
            training_data: None,
            training_labels: None,
        }
    }

    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
        self.training_data = Some(X.clone());
        self.training_labels = Some(y.clone());
    }

    pub fn predict(&self, X: &Array2<f64>) -> Array1<f64> {
        Array1::from(
            X.outer_iter()
                .map(|x| self.predict_sample(&x.view()))
                .collect::<Vec<_>>(),
        )
    }

    fn predict_sample(&self, x: &ndarray::ArrayView1<f64>) -> f64 {
        let mut heap = BinaryHeap::new();
        let training_data = self.training_data.as_ref().unwrap();
        let training_labels = self.training_labels.as_ref().unwrap();

        for (i, train_row) in training_data.outer_iter().enumerate() {
            let distance = self.euclidean_distance(&train_row.view(), x);
            heap.push(Neighbor {
                distance,
                label: training_labels[i],
            });
        }

        let mut neighbors = vec![];
        for _ in 0..self.k {
            if let Some(neighbor) = heap.pop() {
                neighbors.push(neighbor);
            }
        }

        if neighbors.is_empty() {
            return 0.0; // Default value if no neighbors found
        }

        // Use uniform weights for averaging
        let sum: f64 = neighbors.iter().map(|neighbor| neighbor.label).sum();
        let count = neighbors.len() as f64;

        sum / count // Return average
    }

    pub fn euclidean_distance(&self, a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, array};

    #[test]
    fn test_new_knn_regressor() {
        let knn = KNNRegressor::new(3);
        assert_eq!(knn.k, 3);
        assert!(knn.training_data.is_none());
        assert!(knn.training_labels.is_none());
    }

    #[test]
    fn test_fit() {
        let mut knn = KNNRegressor::new(3);
        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.5, 1.5];
        knn.fit(&X, &y);

        assert!(knn.training_data.is_some());
        assert!(knn.training_labels.is_some());
        assert_eq!(knn.training_data.unwrap(), X);
        assert_eq!(knn.training_labels.unwrap(), y);
    }

    #[test]
    fn test_predict() {
        let mut knn = KNNRegressor::new(1);
        let X_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![0.5, 1.5];
        knn.fit(&X_train, &y_train);

        let X_test = array![[1.0, 2.0], [2.0, 3.0]];
        let predictions = knn.predict(&X_test);
        println!("Predictions: {:?}", predictions);
        assert_eq!(predictions, array![0.5, 0.5]);
    }

    #[test]
    fn test_euclidean_distance() {
        let model = KNNRegressor::new(1);
        let array_a = array![1.0, 2.0];
        let a = array_a.view();
        let array_b = array![3.0, 4.0];
        let b = array_b.view();
        let distance = model.euclidean_distance(&a, &b);
        println!("Distance: {:?}", distance);
        assert_eq!(distance, 2.8284271247461903);
    }

    #[test]
    fn test_predict_with_noise() {
        let X = arr2(&[
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ]);
        let y = arr1(&[1.1, 1.9, 3.0, 4.1, 4.9]);

        let mut model = KNNRegressor::new(3);
        model.fit(&X, &y);
        let y_pred = model.predict(&X);
        let actual = arr1(&[2.0, 2.0, 3.0, 4.0, 4.0]);
        println!("Predictions: {:?}", y_pred);
        for (pred, actual) in y_pred.iter().zip(actual.iter()) {
            assert!((pred - actual).abs() < 0.2, "Prediction: {}, Actual: {}", pred, actual);
        }
    }

    #[test]
    fn test_neighbors_ordering() {
        let n1 = Neighbor { distance: 1.0, label: 1.0 };
        let n2 = Neighbor { distance: 2.0, label: 2.0 };

        let mut heap = BinaryHeap::new();
        heap.push(n1);
        heap.push(n2);

        assert_eq!(heap.pop().unwrap().distance, 1.0);
        assert_eq!(heap.pop().unwrap().distance, 2.0);
    }
}
