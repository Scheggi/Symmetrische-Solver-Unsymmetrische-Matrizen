use ndarray::Array2;
use tempfile::NamedTempFile;
use flate2::read::GzDecoder;
use matrix_market_rs::{MtxData, SymInfo};
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;
use std::path::{PathBuf};
use std::fs::{self, File};
use std::io::{Read, Write};

/// Gets the path to the application's cache directory, creating it if it doesn't exist.
fn get_or_create_cache_dir() -> Result<PathBuf> {
    let cache_dir = dirs_next::cache_dir()
        .context("Could not determine cache directory")?
        .join("solver-benchmark"); // Our application's specific cache folder

    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)
            .with_context(|| format!("Failed to create cache directory at {:?}", cache_dir))?;
    }
    Ok(cache_dir)
}

/// Loads a matrix from a local file path or a URL.
pub(crate) fn load_matrix_from_path(path: &str) -> Result<(Array2<f64>, SymInfo)> {
    let file_on_disk = if path.starts_with("http") {
        let cache_dir = get_or_create_cache_dir()?;
        let filename = path.split('/').last().unwrap_or("matrix.mtx.gz");
        let cached_file_path = cache_dir.join(filename);

        if !cached_file_path.exists() {
            let pb = ProgressBar::new_spinner();
            pb.enable_steady_tick(Duration::from_millis(120));
            pb.set_style(
                ProgressStyle::with_template("{spinner:.blue} {msg}")?
                    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
            );
            pb.set_message(format!("Downloading {}...", filename));

            let response_bytes = reqwest::blocking::get(path)?.bytes()?;
            fs::write(&cached_file_path, &response_bytes)?;
            pb.finish_with_message(format!("Downloaded and cached '{}'.", filename));
        } else {
            tracing::info!("Found '{}' in cache, using local copy.", filename);
        }
        cached_file_path
    } else {
        tracing::info!("Reading matrix from local file path: {}", path);
        PathBuf::from(path)
    };
    let mtx_data: MtxData<_, 2> = if file_on_disk.extension().unwrap_or_default() == "gz" {
        // 1. Open the compressed file
        let compressed_file = File::open(&file_on_disk)?;
        // 2. Decompress its contents into memory
        let mut decoder = GzDecoder::new(compressed_file);
        let mut decompressed_bytes = Vec::new();
        decoder.read_to_end(&mut decompressed_bytes)?;

        // 3. Create a NEW temporary file and write the uncompressed data into it
        let mut temp_uncompressed_file = NamedTempFile::new()?;
        temp_uncompressed_file.write_all(&decompressed_bytes)?;

        // 4. Parse from the new, uncompressed temporary file
        MtxData::<f64>::from_file(temp_uncompressed_file.path()).with_context(|| format!("Failed to parse Matrix Market data from temporary file for {}", path))?
    } else {
        // If the file is not compressed, we can parse it directly
        MtxData::from_file(&file_on_disk).with_context(|| format!("Failed to parse Matrix Market file at path: {}", path))?
    };


    let (shape, indices, values, sym_info) = match mtx_data {
        MtxData::Sparse(shape, indices, data, sym) => {
            (shape, indices, data, sym)
        }
        MtxData::Dense(shape, data, sym) => {
            (shape, vec![], data, sym)
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

    tracing::info!(" Loaded {}x{} matrix.", rows, cols);
    Ok((a, sym_info))
}