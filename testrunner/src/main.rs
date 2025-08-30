use anyhow::{Context, Result};
use clap::Parser;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use walkdir::WalkDir;

// --- Data Structures ---

/// A single benchmark result from the solver, to be deserialized from its JSON output.
#[derive(Debug, Deserialize, Serialize, Clone)]
struct SolverResult {
    solver_name: String,
    matrix_name: String,
    matrix_size: usize,
    time_ms: u128,
    iters: String,
    final_rel_res: f64,
}

/// A combined result including resource usage from the `time` command.
#[derive(Debug, Serialize)]
struct EnrichedResult {
    solver_result: Vec<SolverResult>,
    config_file: String,
    cpu_user_time_s: f64,
    cpu_system_time_s: f64,
    max_ram_kb: u64,
}

// --- CLI Definition ---

#[derive(Parser, Debug)]
#[command(name = "testrunner", author = "Gemini AI")]
#[command(version, about = "Runs the solver benchmark suite and gathers performance data.")]
struct Cli {
    /// Directory containing the JSON configuration files to run.
    config_dir: PathBuf,
}

// --- Parsing Logic ---

fn parse_time_output(stderr: &str) -> Result<(f64, f64, u64)> {
    // Regex to capture the key metrics from `time -v` output
    let user_time_re = Regex::new(r"User time \(seconds\): (\d+\.\d+)")?;
    let sys_time_re = Regex::new(r"System time \(seconds\): (\d+\.\d+)")?;
    let ram_re = Regex::new(r"Maximum resident set size \(kbytes\): (\d+)")?;

    let user_time = user_time_re
        .captures(stderr)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<f64>().ok())
        .context("Could not parse user time from `time` output")?;

    let sys_time = sys_time_re
        .captures(stderr)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<f64>().ok())
        .context("Could not parse system time from `time` output")?;

    let max_ram = ram_re
        .captures(stderr)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<u64>().ok())
        .context("Could not parse max RAM from `time` output")?;

    Ok((user_time, sys_time, max_ram))
}

// --- Main Execution Logic ---

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut enriched_results: Vec<EnrichedResult> = Vec::new();

    // Find all .json files in the specified directory
    let config_files = WalkDir::new(&cli.config_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"));

    for entry in config_files {
        let config_path = entry.path();
        let config_path_str = config_path.to_str().unwrap_or("?");
        println!("--- Running test configuration: {} ---", config_path.display());

        // Execute the solver benchmark binary using `cargo run -p` and wrapped by `/usr/bin/time -v`
        let output = Command::new("/usr/bin/time")
            .args([
                "-v",
                "cargo",
                "run",
                "-p", // Specify the package within the workspace
                "sym_solver",
                "--release",
                "--", // Separator for arguments passed to the binary
                config_path_str,
                "--json", // Ensure the solver outputs machine-readable JSON
            ])
            .output()
            .context(format!("Failed to execute benchmark for {}", config_path_str))?;

        if !output.status.success() {
            eprintln!(
                "Benchmark run failed for config '{}'. Stderr:\n{}",
                config_path_str,
                String::from_utf8_lossy(&output.stderr)
            );
            continue;
        }

        // The `time -v` command writes its output to stderr.
        let stderr_output = String::from_utf8_lossy(&output.stderr);
        let (user_time, sys_time, max_ram) = parse_time_output(&stderr_output)
            .with_context(|| format!("Failed to parse time output for config '{}'", config_path_str))?;

        // The solver benchmark writes its JSON output to stdout.
        let solver_json = String::from_utf8_lossy(&output.stdout);
        let solver_result: Vec<SolverResult> = serde_json::from_str(&solver_json)
            .with_context(|| format!("Failed to parse solver JSON output for config '{}'", config_path_str))?;

        enriched_results.push(EnrichedResult {
            solver_result,
            config_file: config_path_str.to_string(),
            cpu_user_time_s: user_time,
            cpu_system_time_s: sys_time,
            max_ram_kb: max_ram,
        });
        println!("--- Finished: {} ---", config_path.display());
    }

    // Save the final, combined results to a file
    let final_report = serde_json::to_string_pretty(&enriched_results)?;
    let output_filename = "final_benchmark_summary.json";
    fs::write(output_filename, final_report)?;

    println!("\n✅ All benchmarks finished. Combined report saved to '{}'", output_filename);

    Ok(())
}