use ndarray::{Array1, Array2, Array3, Array4, Axis, s, Zip};
use ndarray_rand::rand::{thread_rng, Rng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

use super::activation::{ActivationFunction, ReLU, Sigmoid, Softmax, Tanh, LeakyReLU};
use super::layers::{BatchNorm, Conv2D, MaxPool2D};
use super::optimizers::Optimizer;
use super::schedulers::Scheduler;

#[derive(Serialize, Deserialize)]
pub struct CNN {
    conv_layers: Vec<Conv2D>,
    pool_layers: Vec<MaxPool2D>,
    dense_layers: Vec<Array2<f64>>,
    dense_biases: Vec<Array1<f64>>,
    learning_rate: f64,
    dropout_rates: Vec<f64>,
    l2_lambda: f64,
    optimizer: String, // Placeholder for optimizer type
}

impl CNN {
    pub fn new(
        conv_layers: Vec<Conv2D>,
        pool_layers: Vec<MaxPool2D>,
        dense_layer_sizes: Vec<usize>,
        learning_rate: f64,
        dropout_rates: Vec<f64>,
        l2_lambda: f64,
        optimizer: String,
    ) -> Self {
        let mut dense_layers = Vec::new();
        let mut dense_biases = Vec::new();

        // Determine the size of the flattened input after convolution and pooling layers
        let flattened_size = {
            // Assuming input shape is (channels, height, width)
            let input_shape = (1, 3, 3);  // Update this based on your input shape
            let mut current_shape = input_shape;

            for (conv_layer, pool_layer) in conv_layers.iter().zip(&pool_layers) {
                current_shape = conv_layer.output_shape(current_shape);
                current_shape = pool_layer.output_shape(current_shape);
            }

            current_shape.0 * current_shape.1 * current_shape.2
        };

        // The first dense layer should connect to the flattened output of the convolutional layers
        let mut prev_size = flattened_size;
        for &size in &dense_layer_sizes {
            let weights = Array2::random((prev_size, size), Uniform::new(-1.0, 1.0));
            let biases = Array1::zeros(size);
            dense_layers.push(weights);
            dense_biases.push(biases);
            prev_size = size;
        }

        CNN {
            conv_layers,
            pool_layers,
            dense_layers,
            dense_biases,
            learning_rate,
            dropout_rates,
            l2_lambda,
            optimizer,
        }
    }

    fn get_activation_function(name: &str) -> Box<dyn ActivationFunction> {
        match name {
            "relu" => Box::new(ReLU),
            "tanh" => Box::new(Tanh),
            "sigmoid" => Box::new(Sigmoid),
            "leaky_relu" => Box::new(LeakyReLU::new(0.01)),
            "softmax" => Box::new(Softmax),
            _ => Box::new(Sigmoid),
        }
    }

    fn apply_dropout(input: &Array1<f64>, rate: f64) -> Array1<f64> {
        let mut rng = thread_rng();
        input.mapv(|v| if rng.gen::<f64>() < rate { 0.0 } else { v }) / (1.0 - rate)
    }

    pub fn forward(&self, input: &Array3<f64>, training: bool) -> Vec<Array1<f64>> {
        let mut activations = Vec::new();
        let mut current_input = input.clone();

        for (i, conv_layer) in self.conv_layers.iter().enumerate() {
            let conv_output = conv_layer.forward(&current_input);
            let pool_output = self.pool_layers[i].forward(&conv_output);
            current_input = pool_output;
        }

        println!("Shape after conv and pool: {:?}", current_input.shape());

        let flat_input = current_input.clone().into_shape((current_input.len(),)).unwrap();
        let mut dense_input = flat_input;

        println!("Shape after flattening: {:?}", dense_input.shape());

        for (i, (dense_layer, dropout_rate)) in self.dense_layers.iter().zip(&self.dropout_rates).enumerate() {
            println!("Dense layer {} shape: {:?}", i, dense_layer.shape());
            let z = dense_input.dot(dense_layer) + &self.dense_biases[i];
            let activation_func = Self::get_activation_function("relu"); // Replace with your activation function choice
            let mut activation = activation_func.activate(&z);
            if training {
                activation = Self::apply_dropout(&activation, *dropout_rate);
            }
            dense_input = activation.clone();
            activations.push(dense_input.clone());
        }

        activations
    }

    pub fn backward(
        &mut self,
        input: &Array3<f64>,
        activations: &Vec<Array1<f64>>,
        target: &Array1<f64>,
    ) {
        let mut errors = activations.last().unwrap() - target;
        let mut deltas = Vec::new();

        for i in (0..self.dense_layers.len()).rev() {
            let activation_func = Self::get_activation_function("relu"); // Replace with your activation function choice
            let activation = &activations[i];
            let delta = &errors * &activation_func.derivative(activation);
            deltas.push(delta.clone());

            if i > 0 {
                errors = delta.dot(&self.dense_layers[i].t());
            }
        }

        deltas.reverse();
        let flattened_input_size = input.len(); // This should be 9 in your case
        println!("Flattened input size: {}", flattened_input_size); // Debug size
        let mut previous_activation = input.view().into_shape((flattened_input_size,)).unwrap().to_owned();

        for (i, delta) in deltas.iter().enumerate() {
            if i == 0 {
                // For the first layer, we need to reshape previous_activation to match dense_layer[i].nrows
                previous_activation = previous_activation.clone().into_shape((previous_activation.len(),)).unwrap();
            } else {
                previous_activation = activations[i - 1].clone();
            }

            let weight_update = previous_activation.view().insert_axis(Axis(1)).dot(&delta.view().insert_axis(Axis(0)));
            self.dense_layers[i] -= &(weight_update * self.learning_rate);
            self.dense_biases[i] -= &(delta * self.learning_rate);
        }
    }


    pub fn train(&mut self, inputs: &Array4<f64>, targets: &Array2<f64>, epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in inputs.outer_iter().zip(targets.outer_iter()) {
                let input = input.to_owned();
                let target = target.to_owned();
                let activations = self.forward(&input, true);
                self.backward(&input, &activations, &target);
            }
        }
    }

    pub fn predict(&self, input: &Array3<f64>) -> Array1<f64> {
        let activations = self.forward(input, false);
        activations.last().unwrap().clone()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let model: Result<Self, _> = serde_json::from_reader(reader);
        model.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array3, Array4};

    #[test]
    fn test_cnn_forward() {
        let conv_layers = vec![Conv2D::new(1, 1, 2, 1, 0)];
        let pool_layers = vec![MaxPool2D::new(2, 2)];
        let dense_layer_sizes = vec![1, 2]; // Adjusted to match the flattened size of 1
        let learning_rate = 0.01;
        let dropout_rates = vec![0.5, 0.5];
        let l2_lambda = 0.01;
        let optimizer = "sgd".to_string();

        let cnn = CNN::new(
            conv_layers,
            pool_layers,
            dense_layer_sizes,
            learning_rate,
            dropout_rates,
            l2_lambda,
            optimizer,
        );

        let input = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        ];
        println!("Input shape: {:?}", input.shape()); // Debug shape
        let activations = cnn.forward(&input, false);
        assert_eq!(activations.last().unwrap().len(), 2);
    }

    #[test]
    fn test_cnn_train_predict() {
        let conv_layers = vec![Conv2D::new(1, 1, 2, 1, 0)];
        let pool_layers = vec![MaxPool2D::new(2, 2)];
        let dense_layer_sizes = vec![1, 2]; // Adjusted to match the flattened size of 1
        let learning_rate = 0.01;
        let dropout_rates = vec![0.5, 0.5];
        let l2_lambda = 0.01;
        let optimizer = "sgd".to_string();

        let mut cnn = CNN::new(
            conv_layers,
            pool_layers,
            dense_layer_sizes,
            learning_rate,
            dropout_rates,
            l2_lambda,
            optimizer,
        );

        // Shape of inputs should be (num_samples, channels, height, width)
        let inputs = Array4::from_shape_vec(
            (2, 1, 3, 3),
            vec![
                1.0, 2.0, 3.0, 
                4.0, 5.0, 6.0, 
                7.0, 8.0, 9.0,
                9.0, 8.0, 7.0, 
                6.0, 5.0, 4.0, 
                3.0, 2.0, 1.0
            ]
        ).unwrap();
        let targets = array![
            [0.0, 1.0],
            [1.0, 0.0]
        ];

        println!("Inputs shape: {:?}", inputs.shape()); // Debug shape
        println!("Targets shape: {:?}", targets.shape()); // Debug shape

        cnn.train(&inputs, &targets, 1000);

        let prediction = cnn.predict(&array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        ]);
        assert_eq!(prediction.len(), 2);
    }
}