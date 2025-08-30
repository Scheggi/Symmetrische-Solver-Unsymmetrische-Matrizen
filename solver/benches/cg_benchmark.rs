use ndarray::{Array, Array1, Array2};
use criterion::{criterion_group, criterion_main, Criterion};
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

fn cg_benchmark(c: &mut Criterion) {
    // Create a larger, more realistic problem
    let n = 1000; // Problem size: 200x200 matrix
    let max_iter = 100;

    // Create a well-behaved (diagonally dominant) SPD matrix
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

    // The benchmark itself
    c.bench_function("conjugate_gradient_200x200", |bencher| {
        bencher.iter(|| {
            // Use black_box to prevent the compiler from optimizing away the call
            conjugate_gradient(std::hint::black_box(&a), std::hint::black_box(&b), std::hint::black_box(&x0), std::hint::black_box(max_iter))
        })
    });
}

criterion_group!(benches, cg_benchmark);
criterion_main!(benches);