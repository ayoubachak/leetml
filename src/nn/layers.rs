use ndarray::{Array, Array1, Array2, Array3, Array4, Axis, s, Zip};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Conv2D {
    pub filters: Array4<f64>,
    pub biases: Array1<f64>,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2D {
    pub fn new(input_channels: usize, output_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let filters = Array4::random((output_channels, input_channels, kernel_size, kernel_size), Uniform::new(-1.0, 1.0));
        let biases = Array1::zeros(output_channels);

        Conv2D {
            filters,
            biases,
            stride,
            padding,
        }
    }

    pub fn forward(&self, input: &Array3<f64>) -> Array3<f64> {
        let (input_channels, input_height, input_width) = input.dim();
        let output_height = (input_height + 2 * self.padding - self.filters.dim().2) / self.stride + 1;
        let output_width = (input_width + 2 * self.padding - self.filters.dim().3) / self.stride + 1;
        let mut output = Array3::zeros((self.filters.dim().0, output_height, output_width));

        for (filter_index, (filter, &bias)) in self.filters.outer_iter().zip(self.biases.iter()).enumerate() {
            for i in 0..output_height {
                for j in 0..output_width {
                    let h_start = i * self.stride;
                    let w_start = j * self.stride;
                    let patch = input.slice(s![.., h_start..h_start + filter.dim().1, w_start..w_start + filter.dim().2]);
                    let result = Zip::from(&patch).and(&filter).fold(0.0, |acc, &x, &w| acc + x * w);
                    output[(filter_index, i, j)] = result + bias;
                }
            }
        }

        output
    }

    // Add this method
    pub fn output_shape(&self, input_shape: (usize, usize, usize)) -> (usize, usize, usize) {
        let (input_channels, input_height, input_width) = input_shape;
        let output_height = (input_height + 2 * self.padding - self.filters.dim().2) / self.stride + 1;
        let output_width = (input_width + 2 * self.padding - self.filters.dim().3) / self.stride + 1;
        (self.filters.dim().0, output_height, output_width)
    }
}

#[derive(Serialize, Deserialize)]
pub struct MaxPool2D {
    pool_size: usize,
    stride: usize,
}

impl MaxPool2D {
    pub fn new(pool_size: usize, stride: usize) -> Self {
        MaxPool2D { pool_size, stride }
    }

    pub fn forward(&self, input: &Array3<f64>) -> Array3<f64> {
        let (input_channels, input_height, input_width) = input.dim();
        let output_height = (input_height - self.pool_size) / self.stride + 1;
        let output_width = (input_width - self.pool_size) / self.stride + 1;
        let mut output = Array3::zeros((input_channels, output_height, output_width));

        for k in 0..input_channels {
            for i in 0..output_height {
                for j in 0..output_width {
                    let h_start = i * self.stride;
                    let w_start = j * self.stride;
                    let patch = input.slice(s![k, h_start..h_start + self.pool_size, w_start..w_start + self.pool_size]);
                    output[(k, i, j)] = patch.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                }
            }
        }

        output
    }

    // Add this method
    pub fn output_shape(&self, input_shape: (usize, usize, usize)) -> (usize, usize, usize) {
        let (input_channels, input_height, input_width) = input_shape;
        let output_height = (input_height - self.pool_size) / self.stride + 1;
        let output_width = (input_width - self.pool_size) / self.stride + 1;
        (input_channels, output_height, output_width)
    }
}

#[derive(Serialize, Deserialize)]
pub struct BatchNorm {
    gamma: Array1<f64>,
    beta: Array1<f64>,
    running_mean: Array1<f64>,
    running_var: Array1<f64>,
    epsilon: f64,
    momentum: f64,
}

impl BatchNorm {
    pub fn new(num_features: usize, epsilon: f64, momentum: f64) -> Self {
        BatchNorm {
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),
            epsilon,
            momentum,
        }
    }

    pub fn forward(&mut self, input: &Array2<f64>, training: bool) -> Array2<f64> {
        let mean = input.mean_axis(Axis(0)).unwrap();
        let var = input.var_axis(Axis(0), 0.0);
        if training {
            self.running_mean = self.momentum * &self.running_mean + (1.0 - self.momentum) * &mean;
            self.running_var = self.momentum * &self.running_var + (1.0 - self.momentum) * &var;
        }
        let normalized = (input - &mean) / ((&var + self.epsilon).mapv(f64::sqrt));
        &self.gamma * &normalized + &self.beta
    }
}


impl BatchNorm {
    pub fn forward_3d(&mut self, input: &Array3<f64>, training: bool) -> Array3<f64> {
        let (c, h, w) = input.dim();
        let input_reshaped = input.view().into_shape((c, h * w)).unwrap().to_owned(); // Convert the view reference to an owned array reference
        let output_reshaped = self.forward(&input_reshaped, training);
        output_reshaped.into_shape((c, h, w)).unwrap()
    }

}
