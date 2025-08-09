use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;
/// Holds the result of the BiCGSTAB solver.
#[derive(Debug)]
pub struct BicgstabResult {
    /// The solution vector `x`.
    //pub x: Array1<f64>,
    /// `true` if the solution converged within the given tolerance.
    //pub success: bool,
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
) -> Result<BicgstabResult> {
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
        return Ok(BicgstabResult {rel_res: 0.0, iters: 0 }); //x: Array1::zeros(n), success: true, rel_res: 0.0, iters: 0 };
    }

    // ---- Main Iteration Loop ----
    for iter_count in 1..=max_iter {
        // rho = r0' * r;
        let rho = r0.t().dot(&r);
        if rho.abs() < 1e-20 {
            return Err(anyhow!("Solver failed due to breakdown (rho is near zero)"));
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
        if r0_dot_v.abs() < 1e-20 {
            return Err(anyhow!("Solver failed due to breakdown (r0' * v is near zero)"));
        }
        alpha = rho / r0_dot_v;

        // s = r - alpha * v;
        let s = &r - (alpha * &v);

        // t = A * s;
        let t = a.dot(&s);

        // omega = (t' * s) / (t' * t);
        let t_dot_t = t.t().dot(&t);
        if t_dot_t.abs() < 1e-20 {
            return Err(anyhow!("Solver failed due to breakdown (t' * t is near zero)"));
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
            return Ok(BicgstabResult {
                //x,
                //success: true,
                rel_res,
                iters: iter_count,
            });
        }

        // Update for next iteration
        rho_old = rho;
    }

    // ---- Loop finished without converging ----
    let final_rel_res = r.norm_l2() / norm_b;
    Ok(BicgstabResult {
        //x,
        //success: false,
        rel_res: final_rel_res,
        iters: max_iter,
    })
}