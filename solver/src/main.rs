use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::error::Error;
use std::{fs, process};
use std::time::Instant;
use matrix_market_rs::{SymInfo};
use ndarray_linalg::Norm;
use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::{EnvFilter, fmt};
mod solvers;
mod reporting;
mod config;
mod io;
mod errors;

use config::{SolverConfig, TestRun};
use io::load_matrix_from_path;
use reporting::{print_summary, BenchmarkResult};
use crate::errors::AppError;
use crate::solvers::{bicgstab, conjugate_gradient,conjugate_gradient_normal_euqitation ,gmres, solve_lu};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
#[command(name = "solver-benchmark")]
#[command(author = "Antonia Vitt")]
struct Cli {
    /// Path to the JSON configuration file for the benchmark.
    config_path: String,
    #[arg(long)]
    json: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    // This sets up logging and allows the level to be configured
    // via the RUST_LOG environment variable.
    fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    let config_path = &cli.config_path;
    if !std::path::Path::new(config_path).exists() {
        tracing::error!("Configuration file not found at '{}'", config_path);
        eprintln!("\nUsage: cargo run --release -- <path_to_config_json>");
        process::exit(1);
    }
    if !std::path::Path::new(config_path).exists() {
        tracing::error!("Configuration file not found at '{}'", config_path);
        eprintln!("\nUsage: cargo run --release -- <path_to_config_json>");
        process::exit(1);
    }
    tracing::info!("Reading test cases from: {}", config_path);
    // Read and parse the entire config file
    let config_content = fs::read_to_string(config_path).with_context(|| format!("Failed to read configuration file at '{}'", config_path))?;
    let test_runs: Vec<TestRun> = serde_json::from_str(&config_content).map_err(|e| {
        anyhow::anyhow!(AppError::ConfigParseError {
        file_path: config_path.to_string(),
        source: e.into(),
    })
    })?;

    let mut all_results: Vec<BenchmarkResult> = Vec::new();
    // Cache for loaded matrices to avoid re-downloading
    let mut matrix_cache: HashMap<String, (Array2<f64>, SymInfo)> = HashMap::new();

    for run in &test_runs {
        // --- Get matrix from cache or load it ---
        let (a, _symmetry) = match matrix_cache.get(&run.matrix_url) {
            Some(data) => data.clone(),
            None => {
                let (a, sym) = load_matrix_from_path(&run.matrix_url).map_err(|e| {
                    AppError::MatrixLoadError { source_url: run.matrix_url.clone(), source: e }
                })?;
                matrix_cache.insert(run.matrix_url.clone(), (a.clone(), sym));
                (a, sym)
            }
        };

        // --- Setup problem ---
        let n = a.shape()[0];
        let x_true = Array1::from_elem(n, 1.0);
        let b = a.dot(&x_true);
        let x0 = Array1::zeros(n);
        let norm_b = b.norm_l2();

        match &run.solver {
            SolverConfig::LU => {
                tracing::info!("   Running LU Decomposition on {}...", run.matrix_name);
                let start = Instant::now();
                let x = solve_lu(&a, &b);
                let duration = start.elapsed();
                let res = (&b - &a.dot(&x)).norm_l2() / norm_b;
                all_results.push(BenchmarkResult { solver_name: "LU Decomposition".to_string(), matrix_name: run.matrix_name.clone(), matrix_size: n, time_ms: duration.as_millis(), iters: "N/A".to_string(), final_rel_res: res });
            }
            SolverConfig::CG => {
                tracing::info!("   Running Conjugate Gradient on {}...", run.matrix_name);
                let start = Instant::now();
                let x = conjugate_gradient(&a, &b, &x0, 2*n);
                let duration = start.elapsed();
                let res = (&b - &a.dot(&x)).norm_l2() / norm_b;
                all_results.push(BenchmarkResult { solver_name: "Conjugate Gradient".to_string(), matrix_name: run.matrix_name.clone(), matrix_size: n, time_ms: duration.as_millis(), iters: "N/A".to_string(), final_rel_res: res });
            }
            SolverConfig::CGNE => {
                tracing::info!("   Running Conjugate Gradient Normal Equitation on {}...", run.matrix_name);
                let start = Instant::now();
                let x = conjugate_gradient_normal_euqitation(&a, &b, &x0, 2*n);
                let duration = start.elapsed();
                let res = (&b - &a.dot(&x)).norm_l2() / norm_b;
                all_results.push(BenchmarkResult { solver_name: "Conjugate Gradient NE".to_string(), matrix_name: run.matrix_name.clone(), matrix_size: n, time_ms: duration.as_millis(), iters: "N/A".to_string(), final_rel_res: res });
            }
            SolverConfig::GMRES { m} => {
                tracing::info!("   Running GMRES(m={}) on {}...", m, run.matrix_name);
                let start = Instant::now();
                let x = gmres(&a, &b, &x0, *m);
                let duration = start.elapsed();
                let res = (&b - &a.dot(&x)).norm_l2() / norm_b;
                all_results.push(BenchmarkResult { solver_name: format!("GMRES({})", m), matrix_name: run.matrix_name.clone(), matrix_size: n, time_ms: duration.as_millis(), iters: m.to_string(), final_rel_res: res });
            }
            SolverConfig::BiCGSTAB { tol } => {
                tracing::info!("   Running BiCGSTAB on {}...", run.matrix_name);
                let start = Instant::now();
                let result = bicgstab(&a, &b, &x0, 2*n, *tol)
                    .map_err(|e| AppError::SolverFailed {
                        solver_name: "BiCGSTAB".to_string(),
                        matrix_name: run.matrix_name.clone(),
                        reason: e.to_string(),
                    })?;                let duration = start.elapsed();
                all_results.push(BenchmarkResult { solver_name: "BiCGSTAB".to_string(), matrix_name: run.matrix_name.clone(), matrix_size: n, time_ms: duration.as_millis(), iters: result.iters.to_string(), final_rel_res: result.rel_res });
            }
        }
    }

    print_summary(&all_results, cli.json);

    Ok(())
}