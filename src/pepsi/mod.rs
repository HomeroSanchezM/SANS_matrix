//! # `pepsi` — Pepsi-SANS invocation and result parsing
//!
//! This module wraps every interaction with the external **Pepsi-SANS** binary:
//!
//! | Sub-module | Responsibility |
//! |---|---|
//! | [`runner`] | Build the command line, spawn processes in parallel, wait for completion |
//! | [`parser`] | Read the `.dat` output files and extract `(Q, I)` as `f32` vectors |
//!
//! ## Pepsi-SANS command line used by this pipeline
//!
//! ```text
//! Pepsi-SANS <pdb> --hModel 3 --conc 2.5 --d2o <fraction> -o <out.dat> --ms 0.32 --x
//! ```
//!
//! | Flag | Value | Meaning |
//! |---|---|---|
//! | `--hModel 3` | fixed | Hydrogen scattering-length model |
//! | `--conc 2.5` | fixed | Sample concentration (mg/mL) |
//! | `--d2o <f>` | per file | D₂O solvent fraction (0.0 – 1.0) |
//! | `-o <path>` | per file | Output `.dat` file path |
//! | `--ms 0.32` | fixed | Minimum q-spacing (Å⁻¹) |
//! | `--x` | fixed | Extended output (extra columns in `.dat`) |
//!
//! ## Typical call sequence (one pattern iteration)
//!
//! ```text
//! // Build one job per %D₂O step
//! let jobs = runner::build_jobs(&pdb_dir, &simul_dir, &percentages, pattern_str);
//!
//! // Run all jobs in parallel (rayon thread pool)
//! runner::run_all(&jobs, &pepsi_path, n_threads)?;
//!
//! // Parse output files → (Q vector, matrix of I columns)
//! let (q_vec, i_matrix) = parser::parse_simulation_folder(&simul_dir, &percentages)?;
//! ```

pub mod parser;
pub mod runner;

// ─── Shared types ────────────────────────────────────────────────────────────

/// One Pepsi-SANS invocation: links one input PDB to one output `.dat` file
/// and carries the D₂O fraction needed to build the `--d2o` flag.
///
/// A `Vec<PepsiJob>` is built by [`runner::build_jobs`] and consumed by
/// [`runner::run_all`].  After [`runner::run_all`] returns successfully every
/// job's `output_path` exists on disk and can be read by [`parser`].
#[derive(Debug, Clone)]
pub struct PepsiJob {
    /// Absolute path to the input `.pdb` file.
    pub pdb_path: std::path::PathBuf,

    /// Absolute path where Pepsi-SANS should write the `.dat` result.
    pub output_path: std::path::PathBuf,

    /// D₂O solvent fraction in `[0.0, 1.0]`, derived from the integer
    /// percentage (e.g. `30 %` → `0.30`).
    pub d2o_fraction: f32,
}
