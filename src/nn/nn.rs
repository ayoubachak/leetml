use indicatif::ProgressBar;
use itertools::izip;
use ndarray::{Array1, Array2, Axis, array};
use ndarray_rand::rand::{thread_rng, Rng};
use ndarray_rand::{rand::rngs::ThreadRng, rand_distr::Uniform};
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

use super::activation::{ActivationFunction, LeakyReLU, ReLU, Sigmoid, Softmax, Tanh};
use super::layers::{BatchNorm, Conv2D, MaxPool2D};

// Additional modules for optimizers and schedulers


#[derive(Serialize, Deserialize)]
pub struct SimpleNN {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    learning_rate: f64,
    dropout_rate: f64, // Add dropout rate
    l2_lambda: f64, // L2 regularization parameter
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
    bias_hidden: Array1<f64>,
    bias_output: Array1<f64>,
}

impl SimpleNN {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64, dropout_rate: f64, l2_lambda: f64) -> Self {
        let weights_input_hidden = Array2::random((input_size, hidden_size), Uniform::new(-1.0, 1.0));
        let weights_hidden_output = Array2::random((hidden_size, output_size), Uniform::new(-1.0, 1.0));
        let bias_hidden = Array1::zeros(hidden_size);
        let bias_output = Array1::zeros(output_size);

        SimpleNN {
            input_size,
            hidden_size,
            output_size,
            learning_rate,
            dropout_rate, // Initialize dropout rate
            l2_lambda, // Initialize L2 regularization parameter
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
        }
    }

    fn apply_dropout(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut rng = thread_rng();
        input.mapv(|v| if rng.gen::<f64>() < self.dropout_rate { 0.0 } else { v }) / (1.0 - self.dropout_rate)
    }

    fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn sigmoid_derivative(x: &Array1<f64>) -> Array1<f64> {
        x * &(1.0 - x)
    }

    fn softmax(x: &Array1<f64>) -> Array1<f64> {
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps = x.mapv(|v| (v - max).exp());
        let sum = exps.sum();
        exps / sum
    }

    pub fn forward(&self, input: &Array1<f64>, training: bool) -> (Array1<f64>, Array1<f64>) {
        let hidden_input = input.dot(&self.weights_input_hidden) + &self.bias_hidden;
        let mut hidden_output = Self::sigmoid(&hidden_input);
        if training {
            hidden_output = self.apply_dropout(&hidden_output);
        }
        let final_input = hidden_output.dot(&self.weights_hidden_output) + &self.bias_output;
        let final_output = Self::sigmoid(&final_input);
    
        (hidden_output, final_output)
    }

    pub fn backward(
        &mut self,
        input: &Array1<f64>,
        hidden_output: &Array1<f64>,
        final_output: &Array1<f64>,
        target: &Array1<f64>,
    ) {
        let output_errors = final_output - target;
        let output_delta = &output_errors * &Self::sigmoid_derivative(final_output);

        let hidden_errors = output_delta.dot(&self.weights_hidden_output.t());
        let hidden_delta = &hidden_errors * &Self::sigmoid_derivative(hidden_output);

        // Update the weights and biases with L2 regularization
        self.weights_hidden_output = &self.weights_hidden_output * (1.0 - self.learning_rate * self.l2_lambda)
            + &hidden_output.view().insert_axis(Axis(1)).dot(&output_delta.view().insert_axis(Axis(0))) * self.learning_rate;
        self.bias_output = &self.bias_output + &output_delta * self.learning_rate;

        self.weights_input_hidden = &self.weights_input_hidden * (1.0 - self.learning_rate * self.l2_lambda)
            + &input.view().insert_axis(Axis(1)).dot(&hidden_delta.view().insert_axis(Axis(0))) * self.learning_rate;
        self.bias_hidden = &self.bias_hidden + &hidden_delta * self.learning_rate;
    }

    pub fn train(&mut self, inputs: &Array2<f64>, targets: &Array2<f64>, epochs: usize) {
        let progress_bar = ProgressBar::new(epochs as u64);
        for epoch in 0..epochs {
            for (input, target) in inputs.outer_iter().zip(targets.outer_iter()) {
                let input = input.to_owned();
                let target = target.to_owned();
                let (hidden_output, final_output) = self.forward(&input, true);
                self.backward(&input, &hidden_output, &final_output, &target);
            }
            progress_bar.inc(1);
        }
        progress_bar.finish_with_message("Training complete");
    }
    
    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        let (_, final_output) = self.forward(input, false);
        final_output
    }

    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let serialized = serde_json::to_string(self).unwrap();
        std::fs::write(path, serialized)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, std::io::Error> {
        let serialized = std::fs::read_to_string(path)?;
        let nn: SimpleNN = serde_json::from_str(&serialized).unwrap();
        Ok(nn)
    }
}


#[derive(Serialize, Deserialize)]
pub struct DenseLayer {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

#[derive(Serialize, Deserialize)]
pub enum Layer {
    Dense(DenseLayer),
    Conv2D(Conv2D),
    MaxPool2D(MaxPool2D),
    BatchNorm(BatchNorm),
}

#[derive(Serialize, Deserialize)]
pub struct MultiLayerNN {
    layer_sizes: Vec<usize>,
    learning_rate: f64,
    layers: Vec<Layer>,
    activations: Vec<String>,
    dropout_rates: Vec<f64>,
    l2_lambda: f64, // L2 regularization parameter
    optimizer: String, // Placeholder for optimizer type
}

impl MultiLayerNN {
    pub fn new(layer_sizes: Vec<usize>, learning_rate: f64, activations: Vec<String>, dropout_rates: Vec<f64>, l2_lambda: f64, optimizer: String) -> Self {
        assert_eq!(layer_sizes.len() - 1, activations.len());
        assert_eq!(layer_sizes.len() - 1, dropout_rates.len());

        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let weights = Array2::random((layer_sizes[i], layer_sizes[i + 1]), Uniform::new(-1.0, 1.0));
            let biases = Array1::zeros(layer_sizes[i + 1]);
            layers.push(Layer::Dense(DenseLayer { weights, biases }));

        }

        MultiLayerNN {
            layer_sizes,
            learning_rate,
            layers,
            activations,
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
        input.mapv(|v| if rng.gen::<f64>() < rate { 0.0 } else { v })
    }

    pub fn forward(&self, input: &Array1<f64>, training: bool) -> Vec<Array1<f64>> {
        let mut activations = Vec::new();
        let mut current_input = input.clone();

        for (i, (layer, act_name)) in izip!(&self.layers, &self.activations).enumerate() {
            let activation_func = Self::get_activation_function(act_name);
            let mut z = match layer {
                Layer::Dense(DenseLayer { weights, biases }) => current_input.dot(weights) + biases,
                _ => unimplemented!(),
            };
            let mut activation = activation_func.activate(&z);
            if training {
                activation = Self::apply_dropout(&activation, self.dropout_rates[i]);
            }
            activations.push(activation.clone());
            current_input = activation;
        }

        activations
    }

    pub fn backward(
        &mut self,
        input: &Array1<f64>,
        activations: &Vec<Array1<f64>>,
        target: &Array1<f64>,
    ) {
        let mut errors = match self.activations.last().unwrap().as_str() {
            "softmax" => activations.last().unwrap() - target,
            _ => target - activations.last().unwrap(),
        };

        let mut deltas = Vec::new();

        for i in (0..self.layers.len()).rev() {
            let activation = &activations[i];
            let activation_func = Self::get_activation_function(&self.activations[i]);
            let delta = &errors * &activation_func.derivative(activation);
            deltas.push(delta.clone());
            if i > 0 {
                errors = delta.dot(&self.layers[i].weights().unwrap().t());
            }
        }
            
        deltas.reverse();
        let mut previous_activation = input.clone();
        
        for (i, delta) in deltas.iter().enumerate() {
            if let Layer::Dense(DenseLayer { weights, biases }) = &mut self.layers[i] {
                let weight_update = previous_activation.view().insert_axis(Axis(1)).dot(&delta.view().insert_axis(Axis(0)));
                *weights += &(weight_update * self.learning_rate);
                *biases += &(delta * self.learning_rate);
            }
        
            if i < activations.len() - 1 {
                previous_activation = activations[i].clone();
            }
        }
    }

    pub fn train(&mut self, inputs: &Array2<f64>, targets: &Array2<f64>, epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in inputs.outer_iter().zip(targets.outer_iter()) {
                let input = input.to_owned();
                let target = target.to_owned();
                let activations = self.forward(&input, true);
                self.backward(&input, &activations, &target);
            }
        }
    }

    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        let activations = self.forward(input, false);
        activations.last().unwrap().clone()
    }

    // Save the model to a file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }

    // Load the model from a file
    pub fn load(path: &str) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let model: Result<Self, _> = serde_json::from_reader(reader);
        model.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

impl Layer {
    pub fn weights(&self) -> Option<&Array2<f64>> {
        match self {
            Layer::Dense(DenseLayer { weights, .. }) => Some(weights),
            _ => None,
        }
    }
}



#[cfg(test)]
mod tests {
    use std::fs::remove_file;

    use ndarray::{array, Array1, Array2};
    use ndarray_rand::rand;
    use ndarray_rand::rand::Rng;

    use crate::nn::activation::ActivationFunction;
    use crate::nn::activation::LeakyReLU;
    use crate::nn::activation::Softmax;
    use crate::nn::layers::BatchNorm;
    use crate::nn::layers::Conv2D;
    use crate::nn::layers::MaxPool2D;
    use crate::nn::nn::SimpleNN;
    use crate::nn::nn::MultiLayerNN;

    #[test]
    fn test_nn_forward() {
        let nn = SimpleNN::new(2, 2, 1, 0.1, 0.5, 0.01);
        let input = array![0.5, 0.1];
        let (_, output) = nn.forward(&input, false); // Pass false for testing

        // Check if the output is in the expected range (0, 1)
        assert!(output.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_nn_train() {
        let mut nn = SimpleNN::new(2, 2, 1, 0.1, 0.5, 0.01);
        let inputs = array![[0.5, 0.1], [0.9, 0.8], [0.2, 0.4]];
        let targets = array![[0.6], [0.1], [0.4]];
        nn.train(&inputs, &targets, 1000);

        let input = array![0.5, 0.1];
        let prediction = nn.predict(&input);

        // Check if the prediction is in the expected range (0, 1)
        assert!(prediction.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_nn_save_load() {
        let mut nn = SimpleNN::new(2, 2, 1, 0.1, 0.5, 0.01);
        let inputs = array![[0.5, 0.1], [0.9, 0.8], [0.2, 0.4]];
        let targets = array![[0.6], [0.1], [0.4]];
        nn.train(&inputs, &targets, 1000);

        nn.save("nn_model.json").expect("Failed to save model");
        let loaded_nn = SimpleNN::load("nn_model.json").expect("Failed to load model");
        remove_file("nn_model.json").expect("Failed to delete model file");

        let input = array![0.5, 0.1];
        let prediction = loaded_nn.predict(&input);

        // Check if the prediction is in the expected range (0, 1)
        assert!(prediction.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_simple_nn_forward() {
        let nn = SimpleNN::new(3, 5, 2, 0.1, 0.5, 0.01);
        let input = array![0.1, 0.2, 0.3];
        let (_, output) = nn.forward(&input, false); // Pass false for testing
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_simple_nn_train_predict() {
        let mut nn = SimpleNN::new(3, 5, 2, 0.1, 0.5, 0.01);
        let inputs = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let targets = array![[0.0, 1.0], [1.0, 0.0]];

        nn.train(&inputs, &targets, 1000);

        let prediction = nn.predict(&array![0.1, 0.2, 0.3]);
        assert_eq!(prediction.len(), 2);
    }

    #[test]
    fn test_simple_nn_forward_with_dropout() {
        let nn = SimpleNN::new(3, 5, 2, 0.1, 0.5, 0.01);
        let input = array![0.1, 0.2, 0.3];
        let (_, output) = nn.forward(&input, true);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_simple_nn_train_predict_with_dropout() {
        let mut nn = SimpleNN::new(3, 5, 2, 0.1, 0.5, 0.01);
        let inputs = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let targets = array![[0.0, 1.0], [1.0, 0.0]];

        nn.train(&inputs, &targets, 1000);

        let prediction = nn.predict(&array![0.1, 0.2, 0.3]);
        assert_eq!(prediction.len(), 2);
    }

    #[test]
    fn test_simple_nn_train_addition(){
        // Create the neural network with appropriate sizes for input, hidden, and output layers
        let input_size = 2;
        let hidden_size = 5;
        let output_size = 1;
        let learning_rate = 0.01;
        let dropout_rate = 0.0; // No dropout for simplicity
        let l2_lambda = 0.0; // No L2 regularization for simplicity

        let mut nn = SimpleNN::new(input_size, hidden_size, output_size, learning_rate, dropout_rate, l2_lambda);

        // Generate training data for addition
        let mut rng = rand::thread_rng();
        let num_samples = 10000;
        let range = 10.0; // Training on numbers between 0 and 10
        let mut inputs = Array2::zeros((num_samples, input_size));
        let mut targets = Array2::zeros((num_samples, output_size));

        for i in 0..num_samples {
            let a: f64 = rng.gen_range(0.0..range);
            let b: f64 = rng.gen_range(0.0..range);
            inputs[(i, 0)] = a;
            inputs[(i, 1)] = b;
            targets[(i, 0)] = a + b;
        }

        // Train the neural network
        let epochs = 100;
        nn.train(&inputs, &targets, epochs);

        // // Test the trained neural network on a broader range of numbers
        let test_inputs = array![[5.5, 4.5], [7.1, 2.2], [9.9, 0.8], [3.3, 7.7], [10.0, 5.0], [8.5, 1.5]];
        for input in test_inputs.outer_iter() {
            let prediction = nn.predict(&input.to_owned());
            println!("Input: {:?}, Prediction: {:?}", input, prediction);
        }
    }

    #[test]
    fn test_conv2d_forward() {
        let conv = Conv2D::new(1, 1, 2, 1, 0);
        let input = array![
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]
        ];
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[1, 2, 2]);
    }

    #[test]
    fn test_maxpool2d_forward() {
        let pool = MaxPool2D::new(2, 2);
        let input = array![
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0]]
        ];
        let output = pool.forward(&input);
        assert_eq!(output.shape(), &[1, 2, 2]);
    }

    #[test]
    fn test_batchnorm_forward() {
        let mut bn = BatchNorm::new(3, 1e-5, 0.1);
        let input = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ];
        let output = bn.forward(&input, true);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_leaky_relu_forward() {
        let leaky_relu = LeakyReLU::new(0.01);
        let input = array![1.0, -2.0, 3.0];
        let output = leaky_relu.activate(&input);
        assert_eq!(output.shape(), &[3]);
    }

    #[test]
    fn test_multilayer_nn_with_conv_and_pool() {
        let layer_sizes = vec![3, 5, 2];
        let learning_rate = 0.01;
        let activations = vec!["relu".to_string(), "sigmoid".to_string()];
        let dropout_rates = vec![0.2, 0.2];
        let l2_lambda = 0.01;
        let optimizer = "sgd".to_string(); // Placeholder for optimizer
        let mut nn = MultiLayerNN::new(layer_sizes, learning_rate, activations, dropout_rates, l2_lambda, optimizer);

        let inputs = array![[0.5, 0.3, 0.2], [0.6, 0.4, 0.1]];
        let targets = array![[1.0, 0.0], [0.0, 1.0]];
        nn.train(&inputs, &targets, 100);

        let input = array![0.5, 0.3, 0.2];
        let output = nn.predict(&input);
        assert_eq!(output.shape(), &[2]);
    }

    #[test]
    fn test_softmax_activation() {
        let softmax = Softmax;
        let input = array![1.0, 2.0, 3.0];
        let output = softmax.activate(&input);
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_multilayer_nn_with_softmax() {
        let layer_sizes = vec![3, 5, 2];
        let learning_rate = 0.01;
        let activations = vec!["relu".to_string(), "softmax".to_string()];
        let dropout_rates = vec![0.2, 0.2];
        let l2_lambda = 0.01;
        let optimizer = "sgd".to_string(); // Placeholder for optimizer
        let mut nn = MultiLayerNN::new(layer_sizes, learning_rate, activations, dropout_rates, l2_lambda, optimizer);

        let inputs = array![[0.5, 0.3, 0.2], [0.6, 0.4, 0.1]];
        let targets = array![[1.0, 0.0], [0.0, 1.0]];
        nn.train(&inputs, &targets, 100);

        let input = array![0.5, 0.3, 0.2];
        let output = nn.predict(&input);
        assert_eq!(output.shape(), &[2]);
    }

    #[test]
    fn test_multi_layer_nn_forward() {
        let nn = MultiLayerNN::new(
            vec![3, 5, 2],
            0.1,
            vec!["relu".to_string(), "softmax".to_string()],
            vec![0.0, 0.0],
            0.01,
            "sgd".to_string(),
        );
        let input = array![0.1, 0.2, 0.3];
        let activations = nn.forward(&input, false);
        assert_eq!(activations.last().unwrap().len(), 2);
    }

    #[test]
    fn test_multi_layer_nn_train_predict() {
        let mut nn = MultiLayerNN::new(
            vec![3, 5, 2],
            0.1,
            vec!["relu".to_string(), "softmax".to_string()],
            vec![0.0, 0.0],
            0.01,
            "sgd".to_string(),
        );
        let inputs = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let targets = array![[0.0, 1.0], [1.0, 0.0]];

        nn.train(&inputs, &targets, 1000);

        let prediction = nn.predict(&array![0.1, 0.2, 0.3]);
        assert_eq!(prediction.len(), 2);
    }

    #[test]
    fn test_multi_layer_nn_forward_with_dropout() {
        let nn = MultiLayerNN::new(
            vec![3, 5, 2],
            0.1,
            vec!["relu".to_string(), "softmax".to_string()],
            vec![0.5, 0.5],
            0.01,
            "sgd".to_string(),
        );
        let input = array![0.1, 0.2, 0.3];
        let activations = nn.forward(&input, true);
        assert_eq!(activations.last().unwrap().len(), 2);
    }

    #[test]
    fn test_multi_layer_nn_train_predict_with_dropout() {
        let mut nn = MultiLayerNN::new(
            vec![3, 5, 2],
            0.1,
            vec!["relu".to_string(), "softmax".to_string()],
            vec![0.5, 0.5],
            0.01,
            "sgd".to_string(),
        );
        let inputs = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let targets = array![[0.0, 1.0], [1.0, 0.0]];

        nn.train(&inputs, &targets, 1000);

        let prediction = nn.predict(&array![0.1, 0.2, 0.3]);
        assert_eq!(prediction.len(), 2);
    }
}