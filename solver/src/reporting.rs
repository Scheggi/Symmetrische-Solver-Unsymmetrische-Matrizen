use serde::Serialize;

/// Holds the result of a single benchmark run.
#[derive(Debug, Serialize)]
pub struct BenchmarkResult {
    pub solver_name: String,
    pub matrix_name: String,
    pub matrix_size: usize,
    pub time_ms: u128,
    pub iters: String,
    pub final_rel_res: f64,
}

/// Prints the final summary table of all benchmark results.
pub fn print_summary(all_results: &[BenchmarkResult], as_json: bool) {
    if as_json {
        // Serialize the entire vector of results into a JSON string and print it.
        let json_output = serde_json::to_string_pretty(all_results)
            .unwrap_or_else(|e| format!("{{\"error\": \"Failed to serialize results: {}\"}}", e));
        println!("{}", json_output);
    }
    else {
        println!("\n\n--- BENCHMARK SUMMARY ---\n");
        println!(
            "{:<20} | {:<12} | {:<10} | {:<12} | {:<7} | {:<20}",
            "Solver", "Matrix", "Size", "Time (ms)", "Iters", "Final Rel. Residual"
        );
        println!("{}", "-".repeat(115));

        for r in all_results {
            println!(
                "{:<20} | {:<12} | {:<10} | {:<12} | {:<7} | {:.4e}",
                r.solver_name, r.matrix_name, r.matrix_size, r.time_ms, r.iters, r.final_rel_res
            );
        }
    }
}