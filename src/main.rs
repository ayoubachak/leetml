use ndarray::array;
use leetml::models::linear_regression::LinearRegression;
use leetml::preprocessing::StandardScaler;






fn main() {
    // Synthetic dataset
    let x = array![
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0]
    ];

    let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

    // Standardize the dataset
    let mut scaler = StandardScaler::new();
    let x_scaled = scaler.fit_transform(&x);

    // Create and fit the model
    let mut model = LinearRegression::new(1);
    model.fit(&x_scaled, &y);

    // Predict
    let y_pred = model.predict(&x_scaled);
    println!("Predictions:\n{:?}", y_pred);
}
