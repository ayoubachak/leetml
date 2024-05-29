use ndarray::{array, Array1, Array2};
use leetml::models::linear_regression::LinearRegression;
use leetml::preprocessing::StandardScaler;

fn main() {
    // Synthetic dataset
    let X = array![
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0]
    ];

    let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

    // Standardize the dataset
    let mut scaler = StandardScaler::new();
    let X_scaled = scaler.fit_transform(&X);

    // Create and fit the model
    let mut model = LinearRegression::new(1);
    model.fit(&X_scaled, &y);

    // Predict
    let y_pred = model.predict(&X_scaled);
    println!("Predictions:\n{:?}", y_pred);
}
