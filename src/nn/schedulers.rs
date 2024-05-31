use ndarray::Array1;


pub trait Scheduler {
    fn get_learning_rate(&self, epoch: usize) -> f64;
}

pub struct StepDecay {
    pub initial_lr: f64,
    pub drop: f64,
    pub epochs_drop: usize,
}

impl Scheduler for StepDecay {
    fn get_learning_rate(&self, epoch: usize) -> f64 {
        self.initial_lr * (self.drop.powf((epoch / self.epochs_drop) as f64))
    }
}

// Add other schedulers like ExponentialDecay, etc.

pub trait Loss {
    fn loss(&self, y: &Array1<f64>, y_pred: &Array1<f64>) -> f64;
    fn gradient(&self, y: &Array1<f64>, y_pred: &Array1<f64>) -> Array1<f64>;
}

pub struct MeanSquaredError;

impl Loss for MeanSquaredError {
    fn loss(&self, y: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        0.5 * (y - y_pred).mapv(|x| x.powi(2)).sum()
    }

    fn gradient(&self, y: &Array1<f64>, y_pred: &Array1<f64>) -> Array1<f64> {
        y_pred - y
    }
}

// pub struct BinaryCrossEntropy;

// impl Loss for BinaryCrossEntropy {
//     fn loss(&self, y: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
//         -y.dot(&y_pred.mapv(|x| x.ln())) - (1.0 - y).dot(&(&1.0 - *y_pred).mapv(|x| x.ln()))
//     }

//     fn gradient(&self, y: &Array1<f64>, y_pred: &Array1<f64>) -> Array1<f64> {
//         (y_pred - y) / (y_pred * (&1.0 - y_pred))
//     }
// }
