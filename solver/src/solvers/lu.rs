use ndarray::{s, Array1, Array2};


/// Decomposes a square matrix A into lower (L) and upper (U) triangular matrices.
///
/// This function implements the Doolittle algorithm for LU factorization, where L
/// has ones on its diagonal.
///
/// # Arguments
/// * `a` - The square matrix `A` to be factorized.
///
/// # Returns
/// A tuple `(L, U)` containing the lower and upper triangular matrices.
///
/// # Panics
/// Panics if a pivot element `U[k, k]` is zero.
pub fn lu_factorization(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (m, n) = a.dim();

    let mut l = Array2::eye(m);
    let mut u = Array2::zeros((m, n));

    for k in 0..m {
        // Calculate the k-th row of U
        // U(k, k:n) = A(k, k:n) - L(k, 1:k-1) * U(1:k-1, k:n)
        let l_row_slice = l.slice(s![k, 0..k]);
        let u_sub_matrix = u.slice(s![0..k, k..n]);
        let product = l_row_slice.dot(&u_sub_matrix);
        let u_row_to_assign = &a.slice(s![k, k..n]) - &product;
        u.slice_mut(s![k, k..n]).assign(&u_row_to_assign);

        // This is a critical point. If U[k,k] is 0, the decomposition fails.
        if u[[k, k]].abs() < 1e-12 {
            panic!("Matrix is singular or requires pivoting.");
        }

        // Calculate the k-th column of L
        // L(i, k) = (A(i, k) - L(i, 1:k-1) * U(1:k-1, k)) / U(k, k);
        for i in (k + 1)..m {
            let l_row_slice = l.slice(s![i, 0..k]);
            let u_col_slice = u.slice(s![0..k, k]);
            let dot_product = l_row_slice.dot(&u_col_slice);

            l[[i, k]] = (a[[i, k]] - dot_product) / u[[k, k]];
        }
    }

    (l, u)
}
/// Solves the linear system Ax = b using LU factorization.
///
/// # Arguments
/// * `a` - The coefficient matrix `A`.
/// * `b` - The vector `b`.
///
/// # Returns
/// The solution vector `x`.
pub fn solve_lu(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    // Step 1: Decompose A into L and U
    let (l, u) = lu_factorization(a);
    let n = b.len();

    // Step 2: Solve Ly = b using forward substitution
    let mut y = Array1::zeros(n);
    for i in 0..n {
        // y(i) = b(i) - L(i, 1:i-1) * y(1:i-1);
        let dot_product = l.slice(s![i, 0..i]).dot(&y.slice(s![0..i]));
        y[i] = b[i] - dot_product; // Division by L[i,i] is skipped as it's 1
    }

    // Step 3: Solve Ux = y using backward substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        // x(i) = (y(i) - U(i, i+1:end) * x(i+1:end)) / U(i,i);
        let dot_product = u.slice(s![i, i + 1..n]).dot(&x.slice(s![i + 1..n]));
        x[i] = (y[i] - dot_product) / u[[i, i]];
    }

    x
}