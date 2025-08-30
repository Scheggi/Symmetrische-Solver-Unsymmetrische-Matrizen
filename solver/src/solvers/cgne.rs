use ndarray::{Array1, Array2};
/// Solves the linear system Ax = b using the Conjugate Gradient method for normal equitaiton.
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
pub fn conjugate_gradient_normal_euqitation(
    a: &Array2<f64>,
    b: &Array1<f64>,
    x0: &Array1<f64>,
    max_iter: usize,
) -> Array1<f64> {

    //At
    let a_t = a.t();

    //b = At*b
    let b_new = a_t.dot(b);

    // A_new = A.t() * A;
    let a_new = a_t.dot(a);

    // x = x0;
    let mut x = x0.clone();

    // r = b - A * x;
    let mut r = b_new - a_new.dot(&x);

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
        let ap = a_new.dot(&p);

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