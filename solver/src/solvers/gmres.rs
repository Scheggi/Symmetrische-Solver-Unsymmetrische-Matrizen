use ndarray::{s, Array1, Array2};
use ndarray_linalg::{LeastSquaresSvd, Norm};


/// Solves the linear system Ax = b using the GMRES method.
///
/// This implementation uses the Arnoldi iteration to build an orthonormal basis
/// for the Krylov subspace, then solves a small least-squares problem to find
/// the optimal solution within that subspace.
///
/// # Arguments
///
/// * `a` - The coefficient matrix `A` (can be non-symmetric).
/// * `b` - The vector `b`.
/// * `x0` - The initial guess for the solution `x`.
/// * `m` - The restart parameter, defining the size of the Krylov subspace.
///
/// # Returns
///
/// The approximate solution vector `x`.
pub fn gmres(
    a: &Array2<f64>,
    b: &Array1<f64>,
    x0: &Array1<f64>,
    m: usize,
) -> Array1<f64> {
    let n = a.shape()[0];
    let mut j: usize = 0;

    // r = b - A * x0;
    let r = b - a.dot(x0);

    // beta = norm(r);
    let beta = r.norm_l2();

    // V: Orthonormal basis for the Krylov subspace
    let mut v = Array2::zeros((n, m + 1));
    // H: Upper-Hessenberg matrix
    let mut h = Array2::zeros((m + 1, m));

    // V(:,1) = r / beta;
    v.column_mut(0).assign(&(&r / beta));

    // Arnoldi iteration loop to build V and H
    for j_idx in 0..m {
        j = j_idx; // Track the current size of the subspace

        // w = A * V(:,j);
        let mut w = a.dot(&v.column(j));

        // Modified Gram-Schmidt process
        for i in 0..=j {
            // H(i,j) = w' * V(:,i);
            h[[i, j]] = w.t().dot(&v.column(i));
            // w = w - H(i,j) * V(:,i);
            w.scaled_add(-h[[i, j]], &v.column(i));
        }

        // H(j+1,j) = norm(w);
        h[[j + 1, j]] = w.norm_l2();

        // Check for "happy breakdown"
        if h[[j + 1, j]].abs() < 1e-12 {
            break;
        }

        // V(:,j+1) = w / H(j+1,j);
        v.column_mut(j + 1).assign(&(&w / h[[j + 1, j]]));
    }

    // The Arnoldi process may have stopped early at iteration `j`.
    // We need to solve the least-squares problem on the subspace we actually built.
    let final_m = j + 1;
    let h_final = h.slice(s![0..final_m, 0..final_m]);

    // Create the right-hand side for the least-squares problem: beta * e1
    let mut e1 = Array1::zeros(final_m);
    e1[0] = 1.0;
    let rhs = beta * &e1;

    // Solve the least-squares problem: y = H \ rhs
    // This finds y that minimizes || Hy - (beta*e1) ||
    let y = h_final
        .least_squares(&rhs)
        .expect("Least squares solve failed")
        .solution;

    // Update the solution: x = x0 + V * y
    let v_final = v.slice(s![.., 0..final_m]);
    let x = x0 + &v_final.dot(&y);

    x
}