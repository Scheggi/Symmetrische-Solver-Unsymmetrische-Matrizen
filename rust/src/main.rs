use ndarray::{Array, s, array, Array1, Array2};
use std::time::Instant;
use ndarray_linalg::{LeastSquaresSvd, Norm};

/// Solves the linear system Ax = b using the Conjugate Gradient method.
///
/// This function implements the algorithm provided in the pseudocode.
///
/// # Arguments
///
/// * `a` - The matrix A (`Array2<f64>`).
/// * `b` - The vector b (`Array1<f64>`).
/// * `x0` - The initial guess for x (`Array1<f64>`).
/// * `max_iter` - The maximum number of iterations.
///
/// # Returns
///
/// The approximate solution vector x (`Array1<f64>`).
pub fn conjugate_gradient(
    a: &Array2<f64>,
    b: &Array1<f64>,
    x0: &Array1<f64>,
    max_iter: usize,
) -> Array1<f64> {
    // x = x0;
    let mut x = x0.clone();

    // r = b - A * x;
    let mut r = b - a.dot(&x);

    // p = r;
    let mut p = r.clone();

    // Cache the dot product of r' with r
    let mut r_dot_r = r.t().dot(&r);

    // Check for immediate convergence (i.e., if the initial guess was correct)
    if r_dot_r.sqrt() < 1e-10 {
        return x;
    }

    // for i = 1:max_iter
    for _ in 0..max_iter {
        // Calculate the matrix-vector product A * p
        let ap = a.dot(&p);

        // alpha = (r' * r) / (p' * A * p);
        let alpha = r_dot_r / p.t().dot(&ap);

        // x = x + alpha * p;
        // This is an in-place update: x += alpha * p
        x.scaled_add(alpha, &p);

        // r_new = r - alpha * A * p;
        let r_new = &r - &(alpha * &ap);

        // Calculate the dot product for the new residual
        let r_new_dot_r_new = r_new.t().dot(&r_new);

        // Check for convergence
        if r_new_dot_r_new.sqrt() < 1e-10 {
            break;
        }

        // beta = (r_new' * r_new) / (r' * r);
        let beta = r_new_dot_r_new / r_dot_r;

        // p = r_new + beta * p;
        p = &r_new + &(beta * &p);

        // r = r_new;
        r = r_new;
        r_dot_r = r_new_dot_r_new;
    }

    x
}

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
/// Holds the result of the BiCGSTAB solver.
#[derive(Debug)]
pub struct BicgstabResult {
    /// The solution vector `x`.
    pub x: Array1<f64>,
    /// `true` if the solution converged within the given tolerance.
    pub success: bool,
    /// The final relative residual norm `||b - Ax|| / ||b||`.
    pub rel_res: f64,
    /// The number of iterations performed.
    pub iters: usize,
}
/// Solves the linear system Ax = b using the BiCGSTAB method.
///
/// # Arguments
/// * `a` - The coefficient matrix `A` (can be non-symmetric).
/// * `b` - The vector `b`.
/// * `x0` - The initial guess for the solution `x`.
/// * `max_iter` - The maximum number of iterations to perform.
/// * `tol` - The convergence tolerance for the relative residual.
///
/// # Returns
/// A `BicgstabResult` struct containing the solution and convergence information.
pub fn bicgstab(
    a: &Array2<f64>,
    b: &Array1<f64>,
    x0: &Array1<f64>,
    max_iter: usize,
    tol: f64,
) -> BicgstabResult {
    // ---- Initialization ----
    let mut x = x0.clone();
    let mut r = b - a.dot(&x);
    let r0 = r.clone(); // Shadow residual

    let mut rho_old = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;

    let n = b.len();
    let mut v = Array1::zeros(n);
    let mut p = Array1::zeros(n);

    let norm_b = b.norm_l2();
    if norm_b == 0.0 { // Trivial case
        return BicgstabResult { x: Array1::zeros(n), success: true, rel_res: 0.0, iters: 0 };
    }

    // ---- Main Iteration Loop ----
    for iter_count in 1..=max_iter {
        // rho = r0' * r;
        let rho = r0.t().dot(&r);
        if rho.abs() < 1e-12 { // Breakdown
            break;
        }

        // beta = (rho / rho_old) * (alpha / omega);
        let beta = (rho / rho_old) * (alpha / omega);

        // p = r + beta * (p - omega * v);
        let p_update = &p - (omega * &v);
        p = &r + (beta * &p_update);

        // v = A * p;
        v = a.dot(&p);

        // alpha = rho / (r0' * v);
        let r0_dot_v = r0.t().dot(&v);
        if r0_dot_v.abs() < 1e-12 { // Breakdown
            break;
        }
        alpha = rho / r0_dot_v;

        // s = r - alpha * v;
        let s = &r - (alpha * &v);

        // t = A * s;
        let t = a.dot(&s);

        // omega = (t' * s) / (t' * t);
        let t_dot_t = t.t().dot(&t);
        if t_dot_t.abs() < 1e-12 { // Breakdown
            break;
        }
        omega = t.t().dot(&s) / t_dot_t;

        // x = x + alpha * p + omega * s;
        x.scaled_add(alpha, &p);
        x.scaled_add(omega, &s);

        // r = s - omega * t;
        r = &s - (omega * &t);

        // ---- Convergence Check ----
        let rel_res = r.norm_l2() / norm_b;
        if rel_res < tol {
            return BicgstabResult {
                x,
                success: true,
                rel_res,
                iters: iter_count,
            };
        }

        // Update for next iteration
        rho_old = rho;
    }

    // ---- Loop finished without converging ----
    let final_rel_res = r.norm_l2() / norm_b;
    BicgstabResult {
        x,
        success: false,
        rel_res: final_rel_res,
        iters: max_iter,
    }
}

fn main() {
    let n = 1000; // A large problem: 1000x1000 matrix
    let max_iter = 200;

    println!("Setting up a {}x{} problem...", n, n);
    let mut a: Array2<f64> = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i == j {
                a[[i, j]] = 2.0 * n as f64;
            } else {
                a[[i, j]] = 1.0;
            }
        }
    }

    let b: Array1<f64> = Array::from_elem(n, 1.0);
    let x0: Array1<f64> = Array::zeros(n);

    println!("Starting solver...");
    let start = Instant::now();

    // Run the solver
    let _solution = conjugate_gradient(&a, &b, &x0, max_iter);

    let duration = start.elapsed();
    println!("Solver finished in {:?}", duration);
    println!("------------------------------------------------------------------------");


    // Define a system Ax = b
    let a: Array2<f64> = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ];
    // The known solution
    let x_true: Array1<f64> = array![1.0, 0.0, -1.0];

    // Calculate b from A and the known solution
    let b = a.dot(&x_true); // b should be [-2.0, -2.0]

    println!("Solving Ax = b for x...");
    println!("Matrix A:\n{:.1}", a);
    println!("Vector b:\n{:.1}", b);
    println!("--------------------------------------");

    // --- Verification of lu_factorization ---
    let (l, u) = lu_factorization(&a);
    println!("Matrix L:\n{:.3}", l);
    println!("Matrix U:\n{:.3}", u);
    println!("L * U:\n{:.1}", l.dot(&u)); // Should be equal to A
    println!("--------------------------------------");

    // --- Solve the system ---
    let x_solution = solve_lu(&a, &b);

    println!("Solution x found:\n{:.6}", x_solution);
    println!("Expected solution x:\n{:.1}", x_true);
    println!("------------------------------------------------------------------------");

    // Define a non-symmetric system Ax = b
    let a: Array2<f64> = array![
        [1.0, 4.0, 0.0, 1.0],
        [2.0, 1.0, 5.0, 0.0],
        [0.0, 2.0, 1.0, 3.0],
        [1.0, 0.0, 2.0, 1.0]
    ];

    let b: Array1<f64> = array![10.0, 18.0, 14.0, 8.0];

    // An initial guess (can be zeros)
    let x0: Array1<f64> = array![0.0, 0.0, 0.0, 0.0];

    // Set the restart parameter (size of Krylov subspace)
    // For an n x n system, m=n guarantees convergence in one go (like LU)
    // but the power of GMRES is using a smaller m.
    let m = 20;

    println!("Solving Ax = b using GMRES(m={})...", m);
    println!("Matrix A:\n{:.1}", a);
    println!("Vector b:\n{:.1}", b);
    println!("--------------------------------------");

    // Calculate initial residual norm
    let initial_residual_norm = (b.clone() - a.dot(&x0)).norm_l2();
    println!("Initial residual norm ||b - Ax0||: {:.6}", initial_residual_norm);

    // Solve the system
    let x_solution = gmres(&a, &b, &x0, m);

    // Calculate final residual norm
    let final_residual_norm = (b.clone() - a.dot(&x_solution)).norm_l2();

    println!("\nSolution x found:\n{:.6}", x_solution);
    println!("Final residual norm ||b - Ax||:   {:.6}", final_residual_norm);

    // For this specific problem, the exact solution is [1, 2, 3, 1]
    println!("\nExact solution is: [1.0, 2.0, 3.0, 1.0]");
    println!("------------------------------------------------------------------------");

    let a: Array2<f64> = array![
        [1.0, 4.0, 0.0, 1.0],
        [2.0, 1.0, 5.0, 0.0],
        [0.0, 2.0, 1.0, 3.0],
        [1.0, 0.0, 2.0, 1.0]
    ];

    let b: Array1<f64> = array![10.0, 18.0, 14.0, 8.0];

    // An initial guess
    let x0: Array1<f64> = array![0.0, 0.0, 0.0, 0.0];

    let max_iter = 100;
    let tol = 1e-8;

    println!("Solving Ax = b using BiCGSTAB...");
    println!("Max iterations: {}, Tolerance: {}", max_iter, tol);
    println!("--------------------------------------");

    let initial_residual_norm = (b.clone() - a.dot(&x0)).norm_l2();
    println!("Initial residual norm ||b - Ax0||: {:.6}", initial_residual_norm);

    // Solve the system
    let result = bicgstab(&a, &b, &x0, max_iter, tol);

    println!("\nSolver finished.");
    println!("Result details: {:#?}", result); // Using {:#?} for pretty-printing the struct

    // For this specific problem, the exact solution is [1, 2, 3, 1]
    println!("\nExact solution is: [1.0, 2.0, 3.0, 1.0]");
}