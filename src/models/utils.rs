use ndarray::Array2;

pub const EPSILON: f64 = 1e-10;

// Helper function to swap rows of a matrix
pub fn swap_rows(matrix: &mut Array2<f64>, row1: usize, row2: usize) {
    if row1 != row2 {
        for col in 0..matrix.ncols() {
            matrix.swap((row1, col), (row2, col));
        }
    }
}

// Function to perform Gaussian elimination and back substitution to invert a matrix
pub fn invert_matrix(mut matrix: Array2<f64>, epsilon : f64) -> Option<Array2<f64>> {
    let n = matrix.nrows();
    let mut identity = Array2::eye(n);
    if epsilon > 0.0 {
        // Add a small value to the diagonal elements
        // to avoid division by zero
        for i in 0..n {
            matrix[(i, i)] += epsilon;
        }
    }

    for i in 0..n {
        // Debug: Print the matrix and identity matrix at the start of each iteration

        // Find the pivot row
        let mut pivot = i;
        for j in i + 1..n {
            if matrix[(j, i)].abs() > matrix[(pivot, i)].abs() {
                pivot = j;
            }
        }

        if matrix[(pivot, i)].abs() < 1e-10 {
            return None;
        }

        // Swap rows in both matrices
        swap_rows(&mut matrix, i, pivot);
        swap_rows(&mut identity, i, pivot);

        // Normalize the pivot row
        let pivot_value = matrix[(i, i)];
        for j in 0..n {
            matrix[(i, j)] /= pivot_value;
            identity[(i, j)] /= pivot_value;
        }

        // Debug: Print the matrix and identity matrix after normalization

        // Eliminate the current column
        for j in 0..n {
            if j != i {
                let factor = matrix[(j, i)];
                for k in 0..n {
                    matrix[(j, k)] -= factor * matrix[(i, k)];
                    identity[(j, k)] -= factor * identity[(i, k)];
                }
            }
        }

        // Debug: Print the matrix and identity matrix after elimination
    }

    Some(identity)
}