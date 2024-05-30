use ndarray::{Array1, Array2, Axis, s};
use ndarray_rand::rand::Rng;
use ndarray_rand::{rand, RandomExt};
use ndarray_rand::rand_distr::Uniform;


#[derive(Debug)]
pub struct KMeans {
    n_clusters: usize,
    centroids: Option<Array2<f64>>,
    max_iter: usize,
    tol: f64,
}

impl KMeans {
    pub fn new(n_clusters: usize, max_iter: usize, tol: f64) -> Self {
        KMeans {
            n_clusters,
            centroids: None,
            max_iter,
            tol,
        }
    }

    pub fn fit(&mut self, X: &Array2<f64>) {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let mut rng = rand::thread_rng();
        self.centroids = Some(X.select(Axis(0), &self.init_centroids(n_samples)).to_owned());

        for _ in 0..self.max_iter {
            let centroids = self.centroids.as_ref().unwrap();
            let labels = self.assign_labels(X, centroids);
            let new_centroids = self.update_centroids(X, &labels);

            let shift = (&new_centroids - centroids).mapv(|x| x.abs()).sum();
            self.centroids = Some(new_centroids);

            if shift <= self.tol {
                break;
            }
        }
    }

    pub fn predict(&self, X: &Array2<f64>) -> Array1<usize> {
        let centroids = self.centroids.as_ref().unwrap();
        self.assign_labels(X, centroids)
    }

    fn init_centroids(&self, n_samples: usize) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        (0..self.n_clusters).map(|_| rng.gen_range(0..n_samples)).collect()
    }

    fn assign_labels(&self, X: &Array2<f64>, centroids: &Array2<f64>) -> Array1<usize> {
        let mut labels = Array1::zeros(X.nrows());
        for (i, sample) in X.outer_iter().enumerate() {
            let mut min_dist = f64::MAX;
            for (j, centroid) in centroids.outer_iter().enumerate() {
                let dist = (&sample - &centroid).mapv(|x| x.powi(2)).sum();
                if dist < min_dist {
                    min_dist = dist;
                    labels[i] = j;
                }
            }
        }
        labels
    }

    fn update_centroids(&self, X: &Array2<f64>, labels: &Array1<usize>) -> Array2<f64> {
        let mut new_centroids = Array2::<f64>::zeros((self.n_clusters, X.ncols()));
        let mut counts = vec![0; self.n_clusters];
    
        for (i, sample) in X.outer_iter().enumerate() {
            let label = labels[i];
            let new_row = new_centroids.row(label).to_owned() + &sample;
            new_centroids.row_mut(label).assign(&new_row);
            counts[label] += 1;
        }
    
        for (mut centroid, count) in new_centroids.outer_iter_mut().zip(counts) {
            if count > 0 {
                centroid.mapv_inplace(|x| x / count as f64);
            }
        }
        new_centroids
    }
    
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fit_predict() {
        let X = array![
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0]
        ];

        let mut model = KMeans::new(2, 100, 1e-4);
        model.fit(&X);

        let centroids = model.centroids.as_ref().unwrap();
        assert_eq!(centroids.shape(), &[2, 2]);

        let labels = model.predict(&X);
        assert_eq!(labels.len(), X.nrows());
    }

    #[test]
    fn test_convergence() {
        let X = array![
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0],
            [8.0, 2.0],
            [10.0, 2.0]
        ];

        let mut model = KMeans::new(3, 100, 1e-4);
        model.fit(&X);

        let centroids = model.centroids.as_ref().unwrap();
        let labels = model.predict(&X);

        assert_eq!(centroids.shape(), &[3, 2]);
        assert_eq!(labels.len(), X.nrows());
    }

    
}
