use ndarray::{Array1, Array2, Axis, array};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SimpleNN {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    learning_rate: f64,
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
    bias_hidden: Array1<f64>,
    bias_output: Array1<f64>,
}

impl SimpleNN {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let weights_input_hidden = Array2::random((input_size, hidden_size), Uniform::new(-1.0, 1.0));
        let weights_hidden_output = Array2::random((hidden_size, output_size), Uniform::new(-1.0, 1.0));
        let bias_hidden = Array1::zeros(hidden_size);
        let bias_output = Array1::zeros(output_size);

        SimpleNN {
            input_size,
            hidden_size,
            output_size,
            learning_rate,
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
        }
    }

    fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn sigmoid_derivative(x: &Array1<f64>) -> Array1<f64> {
        x * &(1.0 - x)
    }

    pub fn forward(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let hidden_input = input.dot(&self.weights_input_hidden) + &self.bias_hidden;
        let hidden_output = Self::sigmoid(&hidden_input);
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
        let output_errors = target - final_output;
        let output_delta = &output_errors * &Self::sigmoid_derivative(final_output);

        let hidden_errors = output_delta.dot(&self.weights_hidden_output.t());
        let hidden_delta = &hidden_errors * &Self::sigmoid_derivative(hidden_output);

        // Update the weights and biases
        self.weights_hidden_output = &self.weights_hidden_output + &hidden_output.view().insert_axis(Axis(1)).dot(&output_delta.view().insert_axis(Axis(0))) * self.learning_rate;
        self.bias_output = &self.bias_output + &output_delta * self.learning_rate;

        self.weights_input_hidden = &self.weights_input_hidden + &input.view().insert_axis(Axis(1)).dot(&hidden_delta.view().insert_axis(Axis(0))) * self.learning_rate;
        self.bias_hidden = &self.bias_hidden + &hidden_delta * self.learning_rate;
    }

    pub fn train(&mut self, inputs: &Array2<f64>, targets: &Array2<f64>, epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in inputs.outer_iter().zip(targets.outer_iter()) {
                let input = input.to_owned();
                let target = target.to_owned();
                let (hidden_output, final_output) = self.forward(&input);
                self.backward(&input, &hidden_output, &final_output, &target);
            }
        }
    }

    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        let (_, final_output) = self.forward(input);
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




#[cfg(test)]
mod tests {
    use std::fs::remove_file;

    use ndarray::{array, Array1, Array2};

    use crate::nn::nn::SimpleNN;
    

    #[test]
    fn test_nn_forward() {
        let nn = SimpleNN::new(2, 2, 1, 0.1);
        let input = array![0.5, 0.1];
        let (_, output) = nn.forward(&input);

        // Check if the output is in the expected range (0, 1)
        assert!(output.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_nn_train() {
        let mut nn = SimpleNN::new(2, 2, 1, 0.1);
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
        let mut nn = SimpleNN::new(2, 2, 1, 0.1);
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
}