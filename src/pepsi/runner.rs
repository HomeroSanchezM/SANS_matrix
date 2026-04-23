//! # `pepsi::runner` — Parallel Pepsi-SANS execution
//!
//! This module is responsible for **invoking Pepsi-SANS** for every
//! `(pattern, %D₂O)` pair. It mirrors what the reference shell script does
//! with GNU `parallel`, but entirely within Rust using **Rayon**:
//!
//! ```text
//! # Shell equivalent (for reference only):
//! printf '%s\n' "${pdb_files[@]}" \
//!   | parallel -j 150 \
//!     Pepsi-SANS {} --hModel 3 --conc 2.5 --d2o {d2o} -o {out} --ms 0.32 --x
//! ```
//!
//! ## Fixed Pepsi-SANS flags
//!
//! | Flag | Value | Why fixed |
//! |---|---|---|
//! | `--hModel 3` | 3 | Recommended H-scattering model for deuterated proteins |
//! | `--conc 2.5` | 2.5 mg/mL | Standard sample concentration for this pipeline |
//! | `-ms 0.32` | 0.32 Å⁻¹ | Maximum q value; controls the 50-point grid spacing |
//! | `-x` | (flag) | Extended output — writes extra diagnostic columns to `.dat` |
//!
//! ## Parallelism
//!
//! [`run_all`] uses `rayon::iter::IntoParallelRefIterator` to schedule jobs
//! across a Rayon thread pool. Each job spawns a child process with
//! `std::process::Command`. Because Pepsi-SANS is CPU-bound and single-
//! threaded internally, the optimal number of Rayon threads equals the number
//! of physical CPU cores; the caller controls this via the `n_threads`
//! parameter (maps to `rayon::ThreadPoolBuilder::num_threads`).
//!
//! ## Error handling
//!
//! Any individual Pepsi-SANS failure (non-zero exit code, stderr output) is
//! collected and returned as a combined error after all jobs have finished, so
//! a single bad PDB file does not abort the entire batch.

use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;

use super::PepsiJob;

// ─── Fixed Pepsi-SANS parameters ─────────────────────────────────────────────

/// Hydrogen scattering-length model (3 = recommended for deuterated proteins).
const HMODEL: &str = "3";

/// Sample concentration in mg/mL used for all simulations.
const CONC: &str = "2.5";

/// Maximum q value in Å⁻¹;
const MS: &str = "0.32";

/// Number of points
const NS: &str = "50";

// ─── Public API ──────────────────────────────────────────────────────────────

/// Builds the list of [`PepsiJob`]s for one deuteration pattern.
///
/// For each percentage value in `percentages`, one job is created:
/// - `pdb_path`     → `<pdb_dir>/<pct>.pdb`     (written by the `pdb` module)
/// - `output_path`  → `<simul_dir>/<pct>.dat`   (will be written by Pepsi-SANS)
/// - `d2o_fraction` → `pct as f32 / 100.0`
///
/// The function does **not** touch the filesystem; it only builds the data
/// structure that [`run_all`] will consume.
///
/// # Parameters
/// - `pdb_dir`     — directory containing `0.pdb`, `5.pdb`, …, `100.pdb`.
/// - `simul_dir`   — directory where Pepsi-SANS output files will be created.
/// - `percentages` — ordered slice of D₂O percentages (e.g. `[0, 5, 10, …, 100]`).
pub fn build_jobs(
    pdb_dir: &Path,
    simul_dir: &Path,
    percentages: &[u8],
) -> Vec<PepsiJob> {
    percentages
        .iter()
        .map(|&pct| PepsiJob {
            pdb_path:     pdb_dir.join(format!("{pct}.pdb")),
            output_path:  simul_dir.join(format!("{pct}.dat")),
            d2o_fraction: pct as f32 / 100.0,
        })
        .collect()
}

/// Runs all Pepsi-SANS jobs in parallel and waits for every process to finish.
///
/// Jobs are dispatched through a dedicated Rayon thread pool of size
/// `n_threads`. Each thread spawns one Pepsi-SANS child process at a time;
/// because Pepsi-SANS is itself single-threaded, setting `n_threads` to the
/// number of physical cores saturates the machine without over-subscribing.
///
/// # Failure semantics
///
/// The function collects the errors from **all** failed jobs and returns them
/// as a single combined message. This means a bad PDB file in the middle of
/// the batch does not abort the rest — all jobs finish before the error
/// surfaces to the caller.
///
/// # Parameters
/// - `jobs`       — slice of jobs produced by [`build_jobs`].
/// - `pepsi_path` — path to the Pepsi-SANS executable (e.g. `"Pepsi-SANS"` or
///                  `"/opt/Pepsi-SANS-Linux/Pepsi-SANS"`).
/// - `n_threads`  — size of the Rayon thread pool; `0` means "use Rayon's
///                  default" (number of logical CPUs).
///
/// # Errors
///
/// Returns an error if:
/// - The thread pool cannot be built.
/// - One or more Pepsi-SANS invocations exit with a non-zero status.
/// - A child process cannot be spawned (wrong path, missing executable, etc.).
pub fn run_all(jobs: &[PepsiJob], pepsi_path: &Path, n_threads: usize) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }

    // ── Build a dedicated Rayon thread pool ──────────────────────────────────
    // Using a local pool (rather than the global one) lets the caller control
    // concurrency precisely and avoids interfering with other Rayon work.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads) // 0 = Rayon default (num_cpus)
        .build()
        .context("Failed to build Rayon thread pool for Pepsi-SANS")?;

    // ── Run jobs and collect errors ──────────────────────────────────────────
    // `install` runs the closure inside the pool and returns its result.
    let errors: Vec<String> = pool.install(|| {
        jobs.par_iter()
            .filter_map(|job| run_single_job(job, pepsi_path).err())
            .map(|e| e.to_string())
            .collect()
    });

    if errors.is_empty() {
        Ok(())
    } else {
        bail!(
            "{} Pepsi-SANS job(s) failed:\n{}",
            errors.len(),
            errors.join("\n")
        )
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Builds and executes the Pepsi-SANS command for a single job.
///
/// Equivalent shell command:
/// ```text
/// Pepsi-SANS <pdb>                 \
///     --hModel 3                   \
///     --conc   2.5                 \
///     --d2o    <fraction>          \
///     -o       <output.dat>        \
///     -ms     0.32                \
///     -x
/// ```
///
/// The `--d2o` flag is formatted with two decimal places (e.g. `0.30`) to
/// match Pepsi-SANS's expected input format and to reproduce the behaviour of
/// the reference shell script's `awk "BEGIN { printf \"%.2f\", ... }"`.
///
/// # Errors
///
/// Returns an error if the process cannot be spawned, or if Pepsi-SANS exits
/// with a non-zero status code (stderr is captured and included in the error).
fn run_single_job(job: &PepsiJob, pepsi_path: &Path) -> Result<()> {
    // Format D₂O fraction with exactly 2 decimal places.
    // Examples: 0 % → "0.00", 5 % → "0.05", 100 % → "1.00"
    let d2o_str = format!("{:.2}", job.d2o_fraction);

    let output = Command::new(pepsi_path)
        .arg(&job.pdb_path)
        // ── Fixed flags (same for every invocation in this pipeline) ─────────
        .arg("--hModel").arg(HMODEL)
        .arg("--conc").arg(CONC)
        // ── Per-job D₂O fraction ─────────────────────────────────────────────
        .arg("--d2o").arg(&d2o_str)
        // ── Output file path ─────────────────────────────────────────────────
        .arg("-o").arg(&job.output_path)
        // ── Fixed flags continued ─────────────────────────────────────────────
        .arg("-ms").arg(MS)
        .arg("-ns").arg(NS)
        .arg("-x")           // extended output columns
        // Capture stdout/stderr so failures produce useful diagnostics.
        .output()
        .with_context(|| {
            format!(
                "Failed to spawn Pepsi-SANS for '{}'. \
                 Is '{}' in PATH or is the path correct?",
                job.pdb_path.display(),
                pepsi_path.display()
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "Pepsi-SANS failed (exit {}): pdb='{}' d2o={}\nstderr: {}",
            output.status.code().unwrap_or(-1),
            job.pdb_path.display(),
            d2o_str,
            stderr.trim()
        );
    }

    Ok(())
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fake_job(pct: u8) -> PepsiJob {
        PepsiJob {
            pdb_path:     PathBuf::from(format!("/tmp/pdbs/{pct}.pdb")),
            output_path:  PathBuf::from(format!("/tmp/simul/{pct}.dat")),
            d2o_fraction: pct as f32 / 100.0,
        }
    }

    // ── build_jobs ────────────────────────────────────────────────────────────

    #[test]
    fn build_jobs_correct_count() {
        let pcts = vec![0u8, 5, 10, 100];
        let jobs = build_jobs(
            Path::new("/pdb_dir"),
            Path::new("/simul_dir"),
            &pcts,
        );
        assert_eq!(jobs.len(), 4);
    }

    #[test]
    fn build_jobs_paths_correct() {
        let jobs = build_jobs(
            Path::new("/pdb"),
            Path::new("/simul"),
            &[30],
        );
        assert_eq!(jobs[0].pdb_path,    PathBuf::from("/pdb/30.pdb"));
        assert_eq!(jobs[0].output_path, PathBuf::from("/simul/30.dat"));
    }

    #[test]
    fn build_jobs_d2o_fraction_correct() {
        let jobs = build_jobs(Path::new("/p"), Path::new("/s"), &[0, 50, 100]);
        assert!((jobs[0].d2o_fraction - 0.00).abs() < 1e-6);
        assert!((jobs[1].d2o_fraction - 0.50).abs() < 1e-6);
        assert!((jobs[2].d2o_fraction - 1.00).abs() < 1e-6);
    }

    #[test]
    fn build_jobs_empty_percentages() {
        let jobs = build_jobs(Path::new("/p"), Path::new("/s"), &[]);
        assert!(jobs.is_empty());
    }

    // ── d2o formatting ────────────────────────────────────────────────────────

    #[test]
    fn d2o_fraction_formatted_two_decimals() {
        // Verify that the format string used inside run_single_job produces the
        // expected strings (mirrors the awk printf "%.2f" in the shell script).
        let cases: &[(u8, &str)] = &[
            (0,   "0.00"),
            (5,   "0.05"),
            (10,  "0.10"),
            (30,  "0.30"),
            (100, "1.00"),
        ];
        for (pct, expected) in cases {
            let fraction = *pct as f32 / 100.0;
            assert_eq!(format!("{:.2}", fraction), *expected,
                "Wrong formatting for {}%", pct);
        }
    }

    // ── run_all — empty list ───────────────────────────────────────────────────

    #[test]
    fn run_all_empty_jobs_is_ok() {
        // No jobs → should return Ok immediately without touching the filesystem.
        let result = run_all(&[], Path::new("Pepsi-SANS"), 1);
        assert!(result.is_ok());
    }

    // ── run_all — bad executable path ─────────────────────────────────────────

    #[test]
    fn run_all_nonexistent_binary_returns_error() {
        let jobs = vec![fake_job(0)];
        let result = run_all(&jobs, Path::new("/nonexistent/Pepsi-SANS"), 1);
        assert!(result.is_err());
    }
}
