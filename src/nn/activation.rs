use ndarray::{Array1, Array2};

pub trait ActivationFunction {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64>;
    fn derivative(&self, input: &Array1<f64>) -> Array1<f64>;
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn derivative(&self, input: &Array1<f64>) -> Array1<f64> {
        let activated = self.activate(input);
        &activated * &(1.0 - &activated)
    }
}

pub struct ReLU;

impl ActivationFunction for ReLU {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|v| if v > 0.0 { v } else { 0.0 })
    }

    fn derivative(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }
}

pub struct Tanh;

impl ActivationFunction for Tanh {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|v| v.tanh())
    }

    fn derivative(&self, input: &Array1<f64>) -> Array1<f64> {
        let activated = self.activate(input);
        1.0 - activated.mapv(|v| v.powi(2))
    }
}

pub struct Softmax;

impl ActivationFunction for Softmax {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        let max_val = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = input.iter().map(|v| (v - max_val).exp()).sum();
        input.mapv(|v| (v - max_val).exp() / exp_sum)
    }

    fn derivative(&self, input: &Array1<f64>) -> Array1<f64> {
        let activated = self.activate(input);
        let n = activated.len();
        let mut jacobian = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                jacobian[(i, j)] = if i == j {
                    activated[i] * (1.0 - activated[i])
                } else {
                    -activated[i] * activated[j]
                };
            }
        }
        jacobian.row(0).to_owned()
    }
}