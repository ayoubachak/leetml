use ndarray::{array, Array1, Array3};
use serde::{Deserialize, Serialize};

pub trait ActivationFunction {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64>;
    fn derivative(&self, input: &Array1<f64>) -> Array1<f64>;
    fn activate_3d(&self, input: &Array3<f64>) -> Array3<f64> {
        input.mapv(|v| self.activate(&array![v])[0])
    }

    fn derivative_3d(&self, input: &Array3<f64>) -> Array3<f64> {
        input.mapv(|v| self.derivative(&array![v])[0])
    }
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn derivative(&self, input: &Array1<f64>) -> Array1<f64> {
        let sigmoid = self.activate(input);
        sigmoid.mapv(|x| 1.0 - x)
    }
}

pub struct Tanh;

impl ActivationFunction for Tanh {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| x.tanh())
    }

    fn derivative(&self, input: &Array1<f64>) -> Array1<f64> {
        let tanh = self.activate(input);
        1.0 - tanh.mapv(|x| x.powi(2))
    }
}

pub struct ReLU;

impl ActivationFunction for ReLU {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    fn derivative(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}


pub struct Softmax;

impl ActivationFunction for Softmax {
    fn activate(&self, x: &Array1<f64>) -> Array1<f64> {
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps = x.mapv(|v| (v - max).exp());
        let sum = exps.sum();
        exps.mapv(|v| v / sum)
    }

    fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        let softmax_values = self.activate(x);
        softmax_values.mapv(|v| if v > 0.0 { 1.0 - v } else { 0.0 })
    }
}

#[derive(Serialize, Deserialize)]
pub struct LeakyReLU {
    alpha: f64,
}

impl LeakyReLU {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl ActivationFunction for LeakyReLU {
    fn activate(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| if x > 0.0 { x } else { self.alpha * x })
    }

    fn derivative(&self, input: &Array1<f64>) -> Array1<f64> {
        input.mapv(|x| if x > 0.0 { 1.0 } else { self.alpha })
    }
}