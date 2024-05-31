use ndarray::{s, Array1, Array2, Array3, Array4};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::nn::activation::{ActivationFunction, ReLU, Sigmoid, Softmax};
use crate::nn::layers::{BatchNorm, Conv2D, MaxPool2D};
use crate::nn::optimizers::Optimizer;

#[derive(Serialize)]
pub struct CNN {
    conv_layers: Vec<Conv2D>,
    pool_layers: Vec<MaxPool2D>,
    dense_layers: Vec<(Array2<f64>, Array1<f64>)>,
    batch_norm_layers: Vec<BatchNorm>,
    activations: Vec<String>,
    #[serde(skip)] // Skip serialization/deserialization for the optimizer
    optimizer: Box<dyn Optimizer>,
}

impl<'de> Deserialize<'de> for CNN {
    fn deserialize<D>(deserializer: D) -> Result<CNN, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct CNNData {
            conv_layers: Vec<Conv2D>,
            pool_layers: Vec<MaxPool2D>,
            dense_layers: Vec<(Array2<f64>, Array1<f64>)>,
            batch_norm_layers: Vec<BatchNorm>,
            activations: Vec<String>,
        }

        let data = CNNData::deserialize(deserializer)?;

        Ok(CNN {
            conv_layers: data.conv_layers,
            pool_layers: data.pool_layers,
            dense_layers: data.dense_layers,
            batch_norm_layers: data.batch_norm_layers,
            activations: data.activations,
            optimizer: Box::new(crate::nn::optimizers::SGD { learning_rate: 0.01 }), // default optimizer
        })
    }
}

impl CNN {
    pub fn new(
        conv_layers: Vec<Conv2D>,
        pool_layers: Vec<MaxPool2D>,
        dense_layers: Vec<(Array2<f64>, Array1<f64>)>,
        batch_norm_layers: Vec<BatchNorm>,
        activations: Vec<String>,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        CNN {
            conv_layers,
            pool_layers,
            dense_layers,
            batch_norm_layers,
            activations,
            optimizer,
        }
    }

    fn get_activation_function(name: &str) -> Box<dyn ActivationFunction> {
        match name {
            "relu" => Box::new(ReLU),
            "sigmoid" => Box::new(Sigmoid),
            "softmax" => Box::new(Softmax),
            _ => Box::new(Sigmoid),
        }
    }

    pub fn forward(&mut self, input: &Array3<f64>, training: bool) -> Vec<Array3<f64>> {
        let mut activations = Vec::new();
        let mut current_input = input.clone();

        for (i, conv_layer) in self.conv_layers.iter().enumerate() {
            let mut output = conv_layer.forward(&current_input);
            if training {
                output = self.batch_norm_layers[i].forward_3d(&output, training);
            }
            let activation_func = Self::get_activation_function(&self.activations[i]);
            let activation = activation_func.activate_3d(&output);
            activations.push(activation.clone());
            current_input = activation;
        }

        for pool_layer in &self.pool_layers {
            current_input = pool_layer.forward(&current_input);
        }

        let flattened_input = current_input.clone().into_shape((current_input.len(),)).unwrap();
        let mut current_input = flattened_input;

        for (i, (weights, biases)) in self.dense_layers.iter().enumerate() {
            let activation_func = Self::get_activation_function(&self.activations[self.conv_layers.len() + i]);
            let z = current_input.dot(weights) + biases;
            let activation = activation_func.activate(&z);
            current_input = activation.clone();
            activations.push(activation.clone().into_shape((1, 1, activation.len())).unwrap());
        }

        activations
    }

    pub fn backward(&mut self, input: &Array3<f64>, activations: &Vec<Array3<f64>>, target: &Array1<f64>) {
        let mut errors = match self.activations.last().unwrap().as_str() {
            "softmax" => activations.last().unwrap().clone().into_shape(target.raw_dim()).unwrap() - target,
            _ => activations.last().unwrap().clone().into_shape(target.raw_dim()).unwrap() - target,
        };

        let mut deltas = Vec::new();

        for i in (0..self.dense_layers.len()).rev() {
            let activation = &activations[self.conv_layers.len() + i].clone().into_shape(target.raw_dim()).unwrap();
            let activation_func = Self::get_activation_function(&self.activations[self.conv_layers.len() + i]);
            let delta = &errors * &activation_func.derivative(&activation);
            deltas.push(delta.clone());

            let (weights, biases) = &mut self.dense_layers[i];
            let delta_reshaped = delta.clone().into_shape((delta.len(), 1)).unwrap();
            let dw = delta_reshaped.dot(&activations[self.conv_layers.len() + i - 1].clone().into_shape((1, activations[self.conv_layers.len() + i - 1].len())).unwrap());
            self.optimizer.update(weights, biases, &dw, &delta.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0)));

            if i > 0 {
                errors = delta.dot(&weights.t());
            }
        }

        deltas.reverse();
        let mut previous_activation = input.clone();
        let mut gradient_data = Vec::new();

        for (i, delta) in deltas.iter().enumerate() {
            if let Some(conv_layer) = self.conv_layers.get(i) {
                let delta_reshaped = delta.clone().into_shape((
                    conv_layer.filters.dim().0,
                    1,
                    1,
                )).unwrap();
                let gradients = self.compute_conv_gradients(conv_layer, &previous_activation, &delta_reshaped);
                gradient_data.push((i, gradients));
            }

            if i < activations.len() - 1 {
                previous_activation = activations[i].clone();
            }
        }

        for (i, gradients) in gradient_data {
            if let Some(conv_layer) = self.conv_layers.get_mut(i) {
                let filter_shape = conv_layer.filters.dim();
                let mut filters_owned = conv_layer.filters.to_owned(); // Clone the filters to an owned type
                self.optimizer.update(
                    &mut filters_owned.into_shape((filter_shape.0 * filter_shape.1 * filter_shape.2, filter_shape.3)).unwrap(),
                    &mut conv_layer.biases,
                    &gradients.0.into_shape((filter_shape.0 * filter_shape.1 * filter_shape.2, filter_shape.3)).unwrap(),
                    &gradients.1,
                );
            }
        }
    }

    pub fn compute_conv_gradients(&self, conv_layer: &Conv2D, input: &Array3<f64>, delta: &Array3<f64>) -> (Array4<f64>, Array1<f64>) {
        let mut d_filters = Array4::zeros(conv_layer.filters.dim());
        let mut d_biases = Array1::zeros(conv_layer.biases.dim());

        // Compute gradients for filters and biases
        for (filter_index, filter) in conv_layer.filters.outer_iter().enumerate() {
            for i in 0..delta.dim().1 {
                for j in 0..delta.dim().2 {
                    let h_start = i * conv_layer.stride;
                    let w_start = j * conv_layer.stride;
                    let patch = input.slice(s![.., h_start..h_start + filter.dim().1, w_start..w_start + filter.dim().2]);
                    d_filters.slice_mut(s![filter_index, .., .., ..]).assign(&(&patch * delta[(filter_index, i, j)]));
                    d_biases[filter_index] += delta[(filter_index, i, j)];
                }
            }
        }

        (d_filters, d_biases)
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

    pub fn predict(&mut self, input: &Array3<f64>) -> Array1<f64> {
        let activations = self.forward(input, false);
        activations.last().unwrap().clone().into_shape((activations.last().unwrap().len(),)).unwrap()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }

    pub fn load(path: &str, optimizer: Box<dyn Optimizer>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut model: CNN = serde_json::from_reader(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        model.optimizer = optimizer;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array4};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    #[test]
    fn test_cnn_forward() {
        let conv_layer = Conv2D::new(1, 1, 3, 1, 0);
        let pool_layer = MaxPool2D::new(2, 2);
        let dense_layer = (Array2::random((4, 2), Uniform::new(-1.0, 1.0)), Array1::zeros(2));
        let batch_norm_layer = BatchNorm::new(4, 1e-5, 0.1);

        let mut cnn = CNN::new(
            vec![conv_layer],
            vec![pool_layer],
            vec![dense_layer],
            vec![batch_norm_layer],
            vec!["relu".to_string(), "softmax".to_string()],
            Box::new(crate::nn::optimizers::SGD { learning_rate: 0.01 })
        );

        let input = array![[[1.0, 0.0, -1.0], [2.0, 3.0, 0.0], [1.0, -1.0, 2.0]]];
        let output = cnn.forward(&input, false);
        assert_eq!(output.last().unwrap().shape(), &[1, 1, 2]);
    }

    
    // #[test]
    // fn test_cnn_train_predict() {
    //     let conv_layer = Conv2D::new(1, 1, 3, 1, 0);
    //     let pool_layer = MaxPool2D::new(2, 2);
    //     let dense_layer = (Array2::random((4, 2), Uniform::new(-1.0, 1.0)), Array1::zeros(2));
    //     let batch_norm_layer = BatchNorm::new(4, 1e-5, 0.1);

    //     let mut cnn = CNN::new(
    //         vec![conv_layer],
    //         vec![pool_layer],
    //         vec![dense_layer],
    //         vec![batch_norm_layer],
    //         vec!["relu".to_string(), "softmax".to_string()],
    //         Box::new(crate::nn::optimizers::SGD { learning_rate: 0.01 })
    //     );

    //     // Correcting the input array to have four dimensions
    //     let inputs = array![[[[1.0, 0.0, -1.0], [2.0, 3.0, 0.0], [1.0, -1.0, 2.0]]]]; // Now a 4D array
    //     let targets = array![[1.0, 0.0]];
    //     cnn.train(&inputs, &targets, 100);

    //     let input = array![[[1.0, 0.0, -1.0], [2.0, 3.0, 0.0], [1.0, -1.0, 2.0]]];
    //     let prediction = cnn.predict(&input);
    //     assert_eq!(prediction.shape(), &[2]);
    // }
}

