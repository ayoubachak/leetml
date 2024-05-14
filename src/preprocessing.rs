use ndarray::{Array1, Array2, Axis};

pub struct StandardScaler {
    mean: Option<Array1<f64>>,
    std_dev: Option<Array1<f64>>,
}

impl StandardScaler {
    pub fn new() -> Self {
        StandardScaler { mean: None, std_dev: None }
    }

    pub fn fit(&mut self, data: &Array2<f64>) {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let std_dev = data.std_axis(Axis(0), 0.0);
        self.mean = Some(mean);
        self.std_dev = Some(std_dev);
    }

    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean.as_ref().unwrap();
        let std_dev = self.std_dev.as_ref().unwrap();
        let mut scaled_data = data.clone();
        for mut row in scaled_data.rows_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                let mean_value = mean[i];
                *val -= mean_value;
                *val /= std_dev[i];
            }
        }
        scaled_data
    }

    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }
}
