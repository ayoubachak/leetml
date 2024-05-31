use ndarray::{Array1, Array2};

pub trait Optimizer {
    fn update(&mut self, weights: &mut Array2<f64>, biases: &mut Array1<f64>, dw: &Array2<f64>, db: &Array1<f64>);
}


pub struct SGD {
    pub learning_rate: f64,
}

impl Optimizer for SGD {
    fn update(&mut self, weights: &mut Array2<f64>, biases: &mut Array1<f64>, dw: &Array2<f64>, db: &Array1<f64>) {
        *weights -= &(dw * self.learning_rate);
        *biases -= &(db * self.learning_rate);
    }
}

// Add other optimizers like Adam, RMSprop, etc.

pub struct Momentum {
    pub learning_rate: f64,
    pub momentum: f64,
    pub velocity_w: Array2<f64>,
    pub velocity_b: Array1<f64>,
}

impl Optimizer for Momentum {
    fn update(&mut self, weights: &mut Array2<f64>, biases: &mut Array1<f64>, dw: &Array2<f64>, db: &Array1<f64>) {
        self.velocity_w = self.momentum * &self.velocity_w - dw * self.learning_rate;
        self.velocity_b = self.momentum * &self.velocity_b - db * self.learning_rate;
        *weights += &self.velocity_w;
        *biases += &self.velocity_b;
    }
}


pub struct Adagrad {
    pub learning_rate: f64,
    pub epsilon: f64,
    pub cache_w: Array2<f64>,
    pub cache_b: Array1<f64>,
}

impl Optimizer for Adagrad {
    fn update(&mut self, weights: &mut Array2<f64>, biases: &mut Array1<f64>, dw: &Array2<f64>, db: &Array1<f64>) {
        self.cache_w += &dw.mapv(|x| x.powi(2));
        self.cache_b += &db.mapv(|x| x.powi(2));
        *weights -= &(dw / (self.cache_w.mapv(|x| x.sqrt() + self.epsilon)) * self.learning_rate);
        *biases -= &(db / (self.cache_b.mapv(|x| x.sqrt() + self.epsilon)) * self.learning_rate);
    }
}


pub struct RMSprop {
    pub learning_rate: f64,
    pub decay: f64,
    pub epsilon: f64,
    pub cache_w: Array2<f64>,
    pub cache_b: Array1<f64>,
}

impl Optimizer for RMSprop {
    fn update(&mut self, weights: &mut Array2<f64>, biases: &mut Array1<f64>, dw: &Array2<f64>, db: &Array1<f64>) {
        self.cache_w = self.decay * &self.cache_w + (1.0 - self.decay) * &dw.mapv(|x| x.powi(2));
        self.cache_b = self.decay * &self.cache_b + (1.0 - self.decay) * &db.mapv(|x| x.powi(2));
        *weights -= &(dw / (self.cache_w.mapv(|x| x.sqrt() + self.epsilon)) * self.learning_rate);
        *biases -= &(db / (self.cache_b.mapv(|x| x.sqrt() + self.epsilon)) * self.learning_rate);
    }
}


pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub t: usize,
    pub m_w: Array2<f64>,
    pub v_w: Array2<f64>,
    pub m_b: Array1<f64>,
    pub v_b: Array1<f64>,
}


impl Optimizer for Adam {
    fn update(&mut self, weights: &mut Array2<f64>, biases: &mut Array1<f64>, dw: &Array2<f64>, db: &Array1<f64>) {
        self.t += 1;
        self.m_w = self.beta1 * &self.m_w + (1.0 - self.beta1) * dw;
        self.v_w = self.beta2 * &self.v_w + (1.0 - self.beta2) * dw.mapv(|x| x.powi(2));
        let m_w_hat = &self.m_w / (1.0 - self.beta1.powi(self.t as i32));
        let v_w_hat = &self.v_w / (1.0 - self.beta2.powi(self.t as i32));
        *weights -= &(m_w_hat / (v_w_hat.mapv(|x| x.sqrt() + self.epsilon)) * self.learning_rate);

        self.m_b = self.beta1 * &self.m_b + (1.0 - self.beta1) * db;
        self.v_b = self.beta2 * &self.v_b + (1.0 - self.beta2) * db.mapv(|x| x.powi(2));
        let m_b_hat = &self.m_b / (1.0 - self.beta1.powi(self.t as i32));
        let v_b_hat = &self.v_b / (1.0 - self.beta2.powi(self.t as i32));
        *biases -= &(m_b_hat / (v_b_hat.mapv(|x| x.sqrt() + self.epsilon)) * self.learning_rate);
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sgd_update() {
        let mut weights = array![[0.5, -0.2], [0.1, 0.4]];
        let mut biases = array![0.1, -0.1];
        let dw = array![[0.2, -0.1], [0.05, -0.05]];
        let db = array![0.02, -0.02];

        let mut optimizer = SGD { learning_rate: 0.1 };
        optimizer.update(&mut weights, &mut biases, &dw, &db);

        assert_eq!(weights, array![[0.48, -0.19], [0.095, 0.405]]);
        assert_eq!(biases, array![0.098, -0.098]);
    }

    #[test]
    fn test_momentum_update() {
        let mut weights = array![[0.5, -0.2], [0.1, 0.4]];
        let mut biases = array![0.1, -0.1];
        let dw = array![[0.2, -0.1], [0.05, -0.05]];
        let db = array![0.02, -0.02];

        let mut optimizer = Momentum {
            learning_rate: 0.1,
            momentum: 0.9,
            velocity_w: Array2::<f64>::zeros((2, 2)),
            velocity_b: Array1::<f64>::zeros(2),
        };
        optimizer.update(&mut weights, &mut biases, &dw, &db);

        assert_eq!(weights, array![[0.48, -0.19], [0.095, 0.405]]);
        assert_eq!(biases, array![0.098, -0.098]);
    }

    #[test]
    fn test_adagrad_update() {
        let mut weights = array![[0.5, -0.2], [0.1, 0.4]];
        let mut biases = array![0.1, -0.1];
        let dw = array![[0.2, -0.1], [0.05, -0.05]];
        let db = array![0.02, -0.02];

        let mut optimizer = Adagrad {
            learning_rate: 0.1,
            epsilon: 1e-8,
            cache_w: Array2::<f64>::zeros((2, 2)),
            cache_b: Array1::<f64>::zeros(2),
        };
        optimizer.update(&mut weights, &mut biases, &dw, &db);

        assert!(weights[[0, 0]] < 0.5);  // Check that weights decreased
        assert!(biases[0] < 0.1);  // Check that biases decreased
    }

    #[test]
    fn test_rmsprop_update() {
        let mut weights = array![[0.5, -0.2], [0.1, 0.4]];
        let mut biases = array![0.1, -0.1];
        let dw = array![[0.2, -0.1], [0.05, -0.05]];
        let db = array![0.02, -0.02];

        let mut optimizer = RMSprop {
            learning_rate: 0.1,
            decay: 0.9,
            epsilon: 1e-8,
            cache_w: Array2::<f64>::zeros((2, 2)),
            cache_b: Array1::<f64>::zeros(2),
        };
        optimizer.update(&mut weights, &mut biases, &dw, &db);

        assert!(weights[[0, 0]] < 0.5);  // Check that weights decreased
        assert!(biases[0] < 0.1);  // Check that biases decreased
    }

    #[test]
    fn test_adam_update() {
        let mut weights = array![[0.5, -0.2], [0.1, 0.4]];
        let mut biases = array![0.1, -0.1];
        let dw = array![[0.2, -0.1], [0.05, -0.05]];
        let db = array![0.02, -0.02];

        let mut optimizer = Adam {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m_w: Array2::<f64>::zeros((2, 2)),
            v_w: Array2::<f64>::zeros((2, 2)),
            m_b: Array1::<f64>::zeros(2),
            v_b: Array1::<f64>::zeros(2),
        };
        optimizer.update(&mut weights, &mut biases, &dw, &db);

        assert!(weights[[0, 0]] < 0.5);  // Check that weights decreased
        assert!(biases[0] < 0.1);  // Check that biases decreased
    }

    #[test]
    fn test_adam_update_2() {
        let mut weights = array![[0.5, -0.2], [0.1, 0.4]];
        let mut biases = array![0.1, -0.1];
        let dw = array![[0.2, -0.1], [0.05, -0.05]];
        let db = array![0.02, -0.02];
        let mut optimizer = Adam {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 1,
            m_w: array![[0.0, 0.0], [0.0, 0.0]],
            v_w: array![[0.0, 0.0], [0.0, 0.0]],
            m_b: array![0.0, 0.0],
            v_b: array![0.0, 0.0],
        };
        optimizer.update(&mut weights, &mut biases, &dw, &db);
        assert!(weights[[0, 0]] < 0.5);  // Check that weights decreased
        assert!(biases[0] < 0.1);  // Check that biases decreased
    }
}