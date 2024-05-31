pub mod schedulers {
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
}