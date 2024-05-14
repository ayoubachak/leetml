use ndarray::{array, Array1, Array2};
use leetml::models::linear_regression::LinearRegression;

fn main() {
    let x = array![
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0]
    ];

    let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

    let mut model = LinearRegression::new();
    model.fit(&x, &y);

    let y_pred = model.predict(&x);
    println!("Predictions:\n{:?}", y_pred);
}
