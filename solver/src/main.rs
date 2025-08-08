use ndarray::{Array1, Array2};
use std::error::Error;
use std::io::{Read, Write};
use std::time::Instant;
use matrix_market_rs::{MtxData, SymInfo};
use ndarray_linalg::Norm;
use tempfile::NamedTempFile;
use flate2::read::GzDecoder;
#[path = "solver.rs"]
mod solver;
use solver::*;

// Struct to hold the outcome of a single benchmark run
#[derive(Debug)]
struct BenchmarkResult {
    solver_name: String,
    matrix_name: String,
    matrix_size: usize,
    time_ms: u128,
    iters: String, // String to allow for "N/A"
    final_rel_res: f64,
}

// Struct to define a test case
struct TestCase<'a> {
    name: &'a str,
    url: &'a str,
}
/// Loads a matrix from a local file path or a URL.
fn load_matrix_from_path(path: &str) -> Result<(Array2<f64>, SymInfo), Box<dyn Error>> {
    let mtx_data: MtxData<f64> = if path.starts_with("http") {
        print!("\n-> Downloading {}...", path.split('/').last().unwrap_or(""));
        let response = reqwest::blocking::get(path)?.bytes()?;
        let mut temp_file = NamedTempFile::new()?;
        if path.ends_with(".gz") {
            let mut decoder = GzDecoder::new(response.as_ref());
            let mut decompressed_bytes = Vec::new();
            decoder.read_to_end(&mut decompressed_bytes)?;
            temp_file.write_all(&decompressed_bytes)?;
        } else {
            temp_file.write_all(&response)?;
        }
        MtxData::from_file(temp_file.path())?
    } else {
        print!("\n-> Reading {}...", path.split('/').last().unwrap_or(""));
        MtxData::from_file(path)?
    };

    // --- DESTRUCTURE THE CORRECT ENUM VARIANT ---
    let (shape, indices, values, sym_info) = match mtx_data {
        MtxData::Sparse(shape, indices, data, sym) => {
            (shape, indices, data, sym)
        }
        MtxData::Dense(_, _, ..) => {
            return Err("Matrix format was Array, but sparse format is required.".into());
        }
    };

    let (rows, cols) = (shape[0], shape[1]);
    let mut a = Array2::zeros((rows, cols));
    // The indices are a Vec of tuples (row, col)
    for k in 0..values.len() {
        let row = indices[k][0];
        let col = indices[k][1];
        let val = values[k];
        a[[row, col]] = val;
        if sym_info != SymInfo::General && row != col {
            a[[col, row]] = val;
        }
    }

    println!(" Loaded {}x{} matrix.", rows, cols);
    Ok((a, sym_info))
}


fn main() -> Result<(), Box<dyn Error>> {
    let test_cases = vec![
        TestCase {
            name: "FIDAP001",
            url: "https://math.nist.gov/MatrixMarket/data/SPARSKIT/fidap/fidap001.mtx.gz",
        },
        TestCase {
            name: "SHERMAN3",
            url: "https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/sherman/sherman3.mtx.gz",
        },
    ];

    let mut all_results: Vec<BenchmarkResult> = Vec::new();

    for case in &test_cases {
        let (a, _symmetry) = load_matrix_from_path(case.url)?;
        let n = a.shape()[0];

        let x_true = Array1::from_elem(n, 1.0);
        let b = a.dot(&x_true);
        let x0 = Array1::zeros(n);
        let norm_b = b.norm_l2();

        println!("   Running Conjugate Gradient...");
        let start = Instant::now();
        let x = conjugate_gradient(&a, &b, &x0, 2 * n);
        let duration = start.elapsed();
        let res = (&b - &a.dot(&x)).norm_l2() / norm_b;
        all_results.push(BenchmarkResult {
            solver_name: "Conjugate Gradient".to_string(),
            matrix_name: case.name.to_string(),
            matrix_size: n,
            time_ms: duration.as_millis(),
            iters: "N/A".to_string(),
            final_rel_res: res,
        });


        println!("   Running LU Decomposition...");
        let start = Instant::now();
        let x = solve_lu(&a, &b);
        let duration = start.elapsed();
        let res = (&b - &a.dot(&x)).norm_l2() / norm_b;
        all_results.push(BenchmarkResult {
            solver_name: "LU Decomposition".to_string(),
            matrix_name: case.name.to_string(),
            matrix_size: n,
            time_ms: duration.as_millis(),
            iters: "N/A".to_string(),
            final_rel_res: res,
        });

        println!("   Running GMRES...");
        let start = Instant::now();
        let x = gmres(&a, &b, &x0, 50);
        let duration = start.elapsed();
        let res = (&b - &a.dot(&x)).norm_l2() / norm_b;
        all_results.push(BenchmarkResult {
            solver_name: "GMRES(50)".to_string(),
            matrix_name: case.name.to_string(),
            matrix_size: n,
            time_ms: duration.as_millis(),
            iters: "50".to_string(),
            final_rel_res: res,
        });

        println!("   Running BiCGSTAB...");
        let start = Instant::now();
        let result = bicgstab(&a, &b, &x0, 2 * n, 1e-8);
        let duration = start.elapsed();
        all_results.push(BenchmarkResult {
            solver_name: "BiCGSTAB".to_string(),
            matrix_name: case.name.to_string(),
            matrix_size: n,
            time_ms: duration.as_millis(),
            iters: result.iters.to_string(),
            final_rel_res: result.rel_res,
        });
    }

    println!("\n\n--- BENCHMARK SUMMARY ---\n");
    println!(
        "{:<20} | {:<12} | {:<10} | {:<12} | {:<10} | {:<20}",
        "Solver", "Matrix", "Size", "Time (ms)", "Iters", "Final Rel. Residual"
    );
    println!("{}", "-".repeat(95));

    for r in &all_results {
        println!(
            "{:<20} | {:<12} | {:<10} | {:<12} | {:<10} | {:.4e}",
            r.solver_name, r.matrix_name, r.matrix_size, r.time_ms, r.iters, r.final_rel_res
        );
    }

    Ok(())
}