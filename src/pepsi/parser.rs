//! # `pepsi::parser` — Pepsi-SANS `.dat` output parser
//!
//! This module reads the `.dat` files written by Pepsi-SANS and extracts the
//! `Q` and `I` values needed to fill the output matrix.
//!
//! ## `.dat` file format
//!
//! Pepsi-SANS writes a plain-text table. Comment lines begin with `#`; data
//! lines contain whitespace-separated floating-point columns:
//!
//! ```text
//! # Pepsi-SANS output
//! # q         I           Iat         Iev         Ihs   ...
//! 0.000000    1.12105e-01 ...
//! 0.005000    1.11127e-01 ...
//! ...
//! ```
//!
//! The columns relevant to this pipeline are:
//! - **Column 0** (index 0): `q` — scattering vector in Å⁻¹
//! - **Column 1** (index 1): `I` — simulated scattering intensity
//!
//! The extended output flag `--x` adds extra diagnostic columns; these are
//! silently ignored here.
//!
//! ## Parsing strategy (spec §6)
//!
//! Because all simulations for one pattern share the same Q grid, we parse Q
//! only once — from the **first** file (0 % D₂O). Subsequent files contribute
//! only their I column.
//!
//! | File | What is parsed |
//! |---|---|
//! | `0.dat` (first) | Q vector **and** I vector |
//! | `5.dat`, `10.dat`, … | I vector only |
//!
//! This avoids redundant string → float conversions and keeps the hot path
//! minimal.
//!
//! ## Output precision
//!
//! All values are stored as **`f32`** (4 bytes) as specified. The `.dat` files
//! use `f64`-precision text, but the downstream B-spline fitting does not
//! benefit from sub-`f32` precision and the matrix would be twice as large.
//!
//! ## Expected number of data points
//!
//! The pipeline is calibrated for **50 Q points** per curve (controlled by
//! `--ms 0.32` passed to Pepsi-SANS). [`parse_simulation_folder`] validates
//! this and returns an error if any file deviates.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};

/// Expected number of data points (Q values) per curve.
/// Controlled by Pepsi-SANS flag `--ms 0.32`.
pub const EXPECTED_POINTS: usize = 50;

// ─── Public API ──────────────────────────────────────────────────────────────

/// Result of parsing one complete simulation folder.
///
/// Returned by [`parse_simulation_folder`].
pub struct SimulationResult {
    /// Q vector shared by all curves in this pattern. Length = [`EXPECTED_POINTS`].
    pub q: Vec<f32>,

    /// One I column per D₂O percentage, in the same order as `percentages`.
    /// Shape: `percentages.len()` × [`EXPECTED_POINTS`].
    pub intensities: Vec<Vec<f32>>,
}

/// Parses an entire simulation folder produced by one Pepsi-SANS batch.
///
/// The function reads one `.dat` file per entry in `percentages`, in order:
/// - The **first** file yields both `Q` and `I`.
/// - Every subsequent file yields only `I` (Q is assumed identical).
///
/// After parsing, every I vector is validated to have exactly
/// [`EXPECTED_POINTS`] entries. If any file is missing or malformed the
/// function returns an error immediately.
///
/// # Parameters
/// - `simul_dir`   — directory containing `<pct>.dat` files.
/// - `percentages` — ordered list of D₂O percentages (must match the files
///                   written by [`runner::run_all`]).
///
/// # Errors
///
/// Returns an error if any `.dat` file is missing, cannot be read, contains
/// non-numeric data, or has a different number of points than [`EXPECTED_POINTS`].
pub fn parse_simulation_folder(
    simul_dir: &Path,
    percentages: &[u8],
) -> Result<SimulationResult> {
    anyhow::ensure!(
        !percentages.is_empty(),
        "percentages list is empty — nothing to parse"
    );

    let mut q_vec: Option<Vec<f32>> = None;
    let mut intensities: Vec<Vec<f32>> = Vec::with_capacity(percentages.len());

    for (idx, &pct) in percentages.iter().enumerate() {
        let dat_path = simul_dir.join(format!("{pct}.dat"));

        if idx == 0 {
            // First file: parse Q and I together.
            let (q, i) = parse_first_dat(&dat_path)
                .with_context(|| format!("Failed to parse first dat file '{}'", dat_path.display()))?;

            validate_length(&q, &dat_path)?;
            validate_length(&i, &dat_path)?;

            q_vec = Some(q);
            intensities.push(i);
        } else {
            // Subsequent files: parse I only.
            let i = parse_subsequent_dat(&dat_path)
                .with_context(|| format!("Failed to parse dat file '{}'", dat_path.display()))?;

            validate_length(&i, &dat_path)?;
            intensities.push(i);
        }
    }

    Ok(SimulationResult {
        // q_vec is guaranteed Some because percentages is non-empty.
        q: q_vec.unwrap(),
        intensities,
    })
}

/// Parses the **first** `.dat` file of a simulation run.
///
/// Returns `(Q, I)` where both vectors have length [`EXPECTED_POINTS`].
/// Comment lines (starting with `#`) are skipped. The function reads column 0
/// as Q and column 1 as I, ignoring any additional columns written by `--x`.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, contains fewer than 2
/// columns on a data line, or contains non-numeric values.
pub fn parse_first_dat(path: &Path) -> Result<(Vec<f32>, Vec<f32>)> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Cannot read '{}'", path.display()))?;

    let mut q_vec = Vec::with_capacity(EXPECTED_POINTS);
    let mut i_vec = Vec::with_capacity(EXPECTED_POINTS);

    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim();

        // Skip blank lines and comment lines (Pepsi-SANS uses '#').
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let (q, i) = parse_qi_line(line, line_no + 1, path)?;
        q_vec.push(q);
        i_vec.push(i);
    }

    Ok((q_vec, i_vec))
}

/// Parses a `.dat` file that is **not** the first one (Q already known).
///
/// Returns only the I column (column index 1). Comment lines are skipped.
/// All other columns are ignored.
///
/// # Errors
///
/// Same conditions as [`parse_first_dat`].
pub fn parse_subsequent_dat(path: &Path) -> Result<Vec<f32>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Cannot read '{}'", path.display()))?;

    let mut i_vec = Vec::with_capacity(EXPECTED_POINTS);

    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // We still need to split to reach column 1, but Q (column 0) is discarded.
        let (_q, i) = parse_qi_line(line, line_no + 1, path)?;
        i_vec.push(i);
    }

    Ok(i_vec)
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Parses a single data line and returns `(q, i)` as `f32`.
///
/// Pepsi-SANS separates columns with one or more whitespace characters (spaces
/// or tabs). Scientific notation (e.g. `1.12105e-01`) is handled by Rust's
/// standard `str::parse::<f32>()`.
///
/// # Errors
///
/// Returns an error if:
/// - The line has fewer than 2 whitespace-separated tokens.
/// - Either of the first two tokens cannot be parsed as `f32`.
fn parse_qi_line(line: &str, line_no: usize, path: &Path) -> Result<(f32, f32)> {
    let mut cols = line.split_ascii_whitespace();

    // Column 0: Q value.
    let q_str = cols.next().with_context(|| {
        format!(
            "Line {line_no} in '{}' has no columns",
            path.display()
        )
    })?;

    // Column 1: I value.
    let i_str = cols.next().with_context(|| {
        format!(
            "Line {line_no} in '{}' has only one column (expected at least 2)",
            path.display()
        )
    })?;

    let q = q_str.parse::<f32>().with_context(|| {
        format!(
            "Cannot parse Q '{}' on line {line_no} of '{}'",
            q_str,
            path.display()
        )
    })?;

    let i = i_str.parse::<f32>().with_context(|| {
        format!(
            "Cannot parse I '{}' on line {line_no} of '{}'",
            i_str,
            path.display()
        )
    })?;

    Ok((q, i))
}

/// Validates that a parsed vector has exactly [`EXPECTED_POINTS`] entries.
///
/// Returns an error with the file path and actual/expected counts if the
/// length does not match.
fn validate_length(vec: &[f32], path: &Path) -> Result<()> {
    if vec.len() != EXPECTED_POINTS {
        bail!(
            "File '{}' has {} data points (expected {}). \
             Check --ms flag passed to Pepsi-SANS.",
            path.display(),
            vec.len(),
            EXPECTED_POINTS
        );
    }
    Ok(())
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Writes `n` data rows plus a header comment to a temp file.
    /// Returns (temp file, expected Q values, expected I values).
    fn write_dat_file(n_rows: usize) -> (NamedTempFile, Vec<f32>, Vec<f32>) {
        let mut file = NamedTempFile::new().unwrap();
        let mut q_expected = Vec::new();
        let mut i_expected = Vec::new();

        writeln!(file, "# Pepsi-SANS output").unwrap();
        writeln!(file, "# q   I   extra").unwrap();

        for i in 0..n_rows {
            let q = i as f32 * 0.01;
            let intensity = (i as f32 + 1.0) * 0.1;
            writeln!(file, "{:.6}  {:.6e}  ignored_column", q, intensity).unwrap();
            q_expected.push(q);
            i_expected.push(intensity);
        }

        (file, q_expected, i_expected)
    }

    // ── parse_first_dat ───────────────────────────────────────────────────────

    #[test]
    fn parse_first_dat_returns_q_and_i() {
        let (f, q_exp, i_exp) = write_dat_file(3);
        let (q, i) = parse_first_dat(f.path()).unwrap();
        assert_eq!(q.len(), 3);
        assert_eq!(i.len(), 3);
        for j in 0..3 {
            assert!((q[j] - q_exp[j]).abs() < 1e-5, "Q mismatch at {j}");
            assert!((i[j] - i_exp[j]).abs() < 1e-5, "I mismatch at {j}");
        }
    }

    #[test]
    fn parse_first_dat_skips_comments() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "# comment 1").unwrap();
        writeln!(file, "# comment 2").unwrap();
        writeln!(file, "0.005000  1.23456e-01").unwrap();

        let (q, i) = parse_first_dat(file.path()).unwrap();
        assert_eq!(q.len(), 1);
        assert!((q[0] - 0.005).abs() < 1e-5);
        assert!((i[0] - 0.123456).abs() < 1e-4);
    }

    #[test]
    fn parse_first_dat_missing_file_returns_error() {
        assert!(parse_first_dat(Path::new("/no/such/file.dat")).is_err());
    }

    #[test]
    fn parse_first_dat_non_numeric_returns_error() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "NaN  bad_value").unwrap();
        // NaN parses successfully in Rust, so let's use something truly invalid.
        let mut file2 = NamedTempFile::new().unwrap();
        writeln!(file2, "0.001  INVALID").unwrap();
        assert!(parse_first_dat(file2.path()).is_err());
    }

    #[test]
    fn parse_first_dat_one_column_line_returns_error() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "0.001").unwrap();
        assert!(parse_first_dat(file.path()).is_err());
    }

    // ── parse_subsequent_dat ─────────────────────────────────────────────────

    #[test]
    fn parse_subsequent_dat_returns_only_i() {
        let (f, _q_exp, i_exp) = write_dat_file(3);
        let i = parse_subsequent_dat(f.path()).unwrap();
        assert_eq!(i.len(), 3);
        for j in 0..3 {
            assert!((i[j] - i_exp[j]).abs() < 1e-5, "I mismatch at {j}");
        }
    }

    // ── validate_length ───────────────────────────────────────────────────────

    #[test]
    fn validate_length_correct_passes() {
        let v = vec![0.0f32; EXPECTED_POINTS];
        assert!(validate_length(&v, Path::new("test.dat")).is_ok());
    }

    #[test]
    fn validate_length_wrong_count_fails() {
        let v = vec![0.0f32; 10]; // too short
        assert!(validate_length(&v, Path::new("test.dat")).is_err());
    }

    // ── parse_simulation_folder ───────────────────────────────────────────────

    #[test]
    fn parse_simulation_folder_assembles_correctly() {
        let dir = TempDir::new().unwrap();
        let pcts: Vec<u8> = vec![0, 5, 10];

        // Write EXPECTED_POINTS rows per file.
        for &pct in &pcts {
            let path = dir.path().join(format!("{pct}.dat"));
            let mut f = fs::File::create(&path).unwrap();
            writeln!(f, "# header").unwrap();
            for j in 0..EXPECTED_POINTS {
                writeln!(
                    f,
                    "{:.6}  {:.6e}",
                    j as f32 * 0.01,
                    pct as f32 * 0.001 + j as f32 * 0.0001
                )
                .unwrap();
            }
        }

        let result = parse_simulation_folder(dir.path(), &pcts).unwrap();

        assert_eq!(result.q.len(), EXPECTED_POINTS);
        assert_eq!(result.intensities.len(), 3);
        for i_vec in &result.intensities {
            assert_eq!(i_vec.len(), EXPECTED_POINTS);
        }
    }

    #[test]
    fn parse_simulation_folder_empty_percentages_returns_error() {
        let dir = TempDir::new().unwrap();
        assert!(parse_simulation_folder(dir.path(), &[]).is_err());
    }

    #[test]
    fn parse_simulation_folder_missing_dat_returns_error() {
        let dir = TempDir::new().unwrap();
        // No files created → first file (0.dat) will be missing.
        assert!(parse_simulation_folder(dir.path(), &[0, 5]).is_err());
    }
}
