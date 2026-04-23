//! # `main` — CLI entry point and pipeline orchestration
//!
//! This binary ties every module together and implements the full Phase 1
//! pipeline described in the project specification:
//!
//! ```text
//! Reference PDB (parsed once)
//!        │
//!        ▼ for every DeuterationPattern
//! ┌──────────────────────────────────────────────────────────┐
//! │  1. apply_pattern()          → deuterated PdbStructure   │
//! │  2. for every %D₂O step:                                 │
//! │       apply_percentage()     → ready PdbStructure        │
//! │       write_pdb_file()       → pdbs_<pattern>/<pct>.pdb  │
//! │  3. run_all() (Pepsi-SANS)   → simul_<pattern>/<pct>.dat │
//! │  4. parse_simulation_folder()→ (Q, [I₀, I₁, …])         │
//! │  5. writer.write_curve()     → appended to output.bin    │
//! │  6. remove_pattern_dirs()    → temp dirs deleted         │
//! └──────────────────────────────────────────────────────────┘
//!        │
//!        ▼
//! output.bin  — column-major f32 binary matrix
//! ```
//!
//! ## CLI usage
//!
//! ```text
//! sans-pipeline --pdb protein.pdb [OPTIONS]
//!
//! Options:
//!   --pdb       <FILE>         Reference PDB file (required)
//!   --mode      complete|reduced  AA grouping mode [default: reduced]
//!   --d2o-step  <N>            D₂O percentage step, 1-100 [default: 5]
//!   --output    <FILE>         Output binary matrix file [default: output.bin]
//!   --threads   <N>            Rayon thread pool size; 0 = auto [default: 0]
//!   --pepsi     <PATH>         Path to Pepsi-SANS executable [default: Pepsi-SANS]
//!   --work-dir  <DIR>          Directory for temporary pdbs_*/simul_* folders [default: .]
//!   --seed      <N>            RNG seed for reproducible D₂O exchange [default: random]
//! ```
//!
//! ## Memory model
//!
//! At any point in time the process holds in memory:
//! - The reference PDB template (one `PdbStructure`, ~few MB).
//! - One deuterated structure (`apply_pattern` clone, same size).
//! - One ready structure per %D₂O step (sequential, not simultaneous).
//! - The `MatrixWriter`'s 8 KB I/O buffer.
//!
//! The output matrix itself is never in memory — it is streamed to disk one
//! 200-byte column at a time.
//!
//! ## Cargo.toml dependencies used
//!
//! ```toml
//! [dependencies]
//! anyhow  = "1"
//! clap    = { version = "4", features = ["derive"] }
//! pdbrust = "0.7"
//! rand    = "0.8"
//! rayon   = "1"
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;

// ── Internal modules (declared in lib.rs or as inline mod in a bin crate) ────
mod amino_acids;
mod binary_vectors;
mod cleanup;
mod output;
mod pdb;
mod pepsi;

use amino_acids::AaMode;
use binary_vectors::PatternIter;
use output::MatrixWriter;
use pepsi::parser::EXPECTED_POINTS;

// ─── CLI definition ───────────────────────────────────────────────────────────

/// SANS deuteration space explorer — Phase 1 curve generator.
///
/// Generates every possible SANS scattering curve for a protein by exhaustively
/// combining all amino-acid deuteration patterns with every D₂O solvent
/// percentage, writing the result as a compact binary f32 matrix.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Reference PDB file to use as the structural template.
    #[arg(long)]
    pdb: PathBuf,

    /// Amino-acid grouping mode.
    /// `reduced`  → 11 groups, 2 048 patterns (fast, ~minutes).
    /// `complete` → 18 groups, 262 144 patterns (slow, ~hours).
    #[arg(long, default_value = "reduced")]
    mode: AaMode,

    /// D₂O percentage step (1–100).
    /// Default 5 produces the grid [0, 5, 10, …, 100] (21 values per pattern).
    #[arg(long, default_value_t = 5)]
    d2o_step: u8,

    /// Output binary matrix file.
    /// Layout: column-major f32 — column 0 is Q, columns 1..N are I curves.
    /// Read in Python: `np.fromfile(path, dtype=np.float32).reshape(-1, 50)`
    #[arg(long, default_value = "output.bin")]
    output: PathBuf,

    /// Rayon thread-pool size for parallel Pepsi-SANS invocations.
    /// 0 means "use all available CPU cores" (Rayon default).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Path to the Pepsi-SANS executable.
    /// Can be a bare name if it is on PATH, or an absolute/relative path.
    #[arg(long, default_value = "Pepsi-SANS")]
    pepsi: PathBuf,

    /// Base directory where temporary `pdbs_*/` and `simul_*/` folders are
    /// created and destroyed. Defaults to the current working directory.
    /// Needs enough free space for one pattern's PDBs + DATs at a time
    /// (~few MB at most).
    #[arg(long, default_value = ".")]
    work_dir: PathBuf,

    /// Random seed for the probabilistic D₂O labile-H selection.
    /// Omit for a non-reproducible run using the OS entropy source.
    /// Provide a fixed integer (e.g. `--seed 42`) for reproducible output.
    #[arg(long)]
    seed: Option<u64>,

    /// Test mode: process only the FIRST pattern (all D2O steps), then stop.
    /// The partial matrix (Q column + n_d2o I columns) is written and finalised
    /// so it can be loaded by Phase 2 code for validation.
    /// Use together with --keep-files to inspect the raw PDB and .dat outputs.
    #[arg(long, default_value_t = false)]
    test_mode: bool,

    /// Keep temporary pdbs_<pattern>/ and simul_<pattern>/ directories
    /// after the run instead of deleting them.
    /// Useful with --test-mode to inspect the PDB files sent to Pepsi-SANS
    /// and the raw .dat outputs it produced.
    #[arg(long, default_value_t = false)]
    keep_files: bool,
}

// ─── Entry point ─────────────────────────────────────────────────────────────

fn main() {
    // Parse CLI; clap prints help/errors and exits automatically on failure.
    let cli = Cli::parse();

    // Delegate to the fallible main so we can use `?` throughout.
    if let Err(e) = run(cli) {
        // Print the full error chain (anyhow includes all context levels).
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}

// ─── Fallible pipeline ────────────────────────────────────────────────────────

/// Runs the full Phase 1 pipeline and returns an error if anything fails.
///
/// Structured as a separate function so `main` can handle the error uniformly
/// and `?` can be used freely inside.
fn run(cli: Cli) -> Result<()> {
    let wall_start = Instant::now();

    // ── 0. Validate and print configuration ──────────────────────────────────
    print_config(&cli);

    anyhow::ensure!(
        cli.d2o_step > 0 && cli.d2o_step <= 100,
        "--d2o-step must be between 1 and 100 (got {})",
        cli.d2o_step
    );
    anyhow::ensure!(
        cli.pdb.exists(),
        "PDB file '{}' does not exist",
        cli.pdb.display()
    );
    anyhow::ensure!(
        cli.work_dir.is_dir() || cli.work_dir == PathBuf::from("."),
        "--work-dir '{}' is not a directory",
        cli.work_dir.display()
    );

    // ── 1. Parse the reference PDB — once ────────────────────────────────────
    // This is the immutable template cloned for every deuteration pattern.
    let template = pdb::parser::load_template(&cli.pdb)
        .with_context(|| format!("Failed to load PDB '{}'", cli.pdb.display()))?;

    // ── 2. Build the D₂O percentage grid ─────────────────────────────────────
    // e.g. step=5  → [0, 5, 10, …, 100]  (21 values)
    //      step=10 → [0, 10, 20, …, 100] (11 values)
    let percentages = pdb::d2o::generate_all_percentages(cli.d2o_step)
        .context("Invalid --d2o-step value")?;
    let n_d2o = percentages.len() as u64;

    // ── 3. Compute total curve count for the matrix writer ────────────────────
    // In test mode we run exactly one pattern, so the matrix holds only
    // n_d2o I columns (plus the Q column). The MatrixWriter validates the
    // final count, so we must pass the correct expected value here.
    let n_patterns = cli.mode.n_patterns();
    let n_curves_total = if cli.test_mode { n_d2o } else { n_patterns * n_d2o };

    if cli.test_mode {
        eprintln!("[main] *** TEST MODE — 1 pattern, {} D2O steps ***", n_d2o);
    } else {
        eprintln!(
            "[main] Patterns: {n_patterns}  ×  D2O steps: {n_d2o}  =  {} curves total",
            n_patterns * n_d2o
        );
    }
    eprintln!(
        "[main] Output matrix size: ~{:.1} MB",
        // +1 for the Q column; each column is EXPECTED_POINTS × 4 bytes
        (1 + n_curves_total) as f64 * EXPECTED_POINTS as f64 * 4.0 / 1_048_576.0
    );
    if cli.keep_files {
        eprintln!("[main] --keep-files active: temp directories will NOT be deleted");
    }

    // ── 4. Initialise the output matrix writer ────────────────────────────────
    // The writer opens the output file immediately and streams columns to it.
    // Memory usage: O(1) — only an 8 KB write buffer.
    let mut writer = MatrixWriter::new(&cli.output, n_curves_total, EXPECTED_POINTS)
        .with_context(|| format!("Cannot create output file '{}'", cli.output.display()))?;

    // Tracks whether the Q vector has been written yet (done once, from the
    // first successfully parsed simulation).
    let mut q_written = false;

    // ── 5. Initialise the RNG for probabilistic D₂O exchange ─────────────────
    // A single RNG is advanced across all (pattern, pct) calls so there is no
    // correlation between consecutive patterns.
    let mut rng: StdRng = match cli.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None    => StdRng::from_entropy(),
    };

    // ── 6. Main loop: iterate over every deuteration pattern ─────────────────
    let mut pattern_idx: u64 = 0;

    for pattern in PatternIter::new(cli.mode) {
        pattern_idx += 1;
        let pattern_str = pattern.to_pattern_string();

        // ── Per-pattern wall-clock timer ──────────────────────────────────────
        // Measures the full cost of one iteration: PDB writing, Pepsi-SANS,
        // .dat parsing, and I/O to the output matrix.
        let pattern_start = Instant::now();

        // Progress report every 100 patterns (or always in debug/test mode).
        if pattern_idx % 100 == 1 || cfg!(debug_assertions) || cli.test_mode {
            eprintln!(
                "[main] Pattern {pattern_idx}/{n_patterns}  ({pattern_str})"
            );
        }

        // Paths for this pattern's temporary directories.
        let pdb_dir   = cleanup::pdb_dir(&cli.work_dir, &pattern_str);
        let simul_dir = cleanup::simul_dir(&cli.work_dir, &pattern_str);

        // Run this pattern inside a helper that returns an error on failure.
        // On error we still attempt cleanup (unless --keep-files) before propagating.
        let result = run_pattern(
            &pattern,
            cli.mode,
            &template,
            &percentages,
            &pdb_dir,
            &simul_dir,
            &cli.pepsi,
            cli.threads,
            &mut rng,
        );

        // Cleanup: skip if --keep-files is set so the user can inspect the
        // raw PDB and .dat files. In normal runs always clean up, even on error,
        // to avoid exhausting disk space over thousands of patterns.
        if !cli.keep_files {
            cleanup::remove_pattern_dirs(&cli.work_dir, &pattern_str);
        } else {
            eprintln!(
                "[main] Kept:  {}  and  {}",
                pdb_dir.display(),
                simul_dir.display()
            );
        }

        // Now propagate any pattern-level error.
        let sim_result = result.with_context(|| {
            format!("Pattern {pattern_idx}/{n_patterns} ({pattern_str}) failed")
        })?;

        // ── Write Q column (once, from the very first pattern) ────────────────
        if !q_written {
            writer.write_q(&sim_result.q).context("Failed to write Q column")?;
            q_written = true;
        }

        // ── Append all I columns for this pattern ─────────────────────────────
        for i_vec in &sim_result.intensities {
            writer.write_curve(i_vec).with_context(|| {
                format!("Failed to write I curve for pattern {pattern_str}")
            })?;
        }

        // ── Per-pattern timing report ─────────────────────────────────────────
        let pattern_elapsed = pattern_start.elapsed();
        eprintln!(
            "[timer] Pattern {pattern_idx}/{n_patterns} ({pattern_str}): {:.3}s               (D2O steps: {}, curves written: {})",
            pattern_elapsed.as_secs_f64(),
            n_d2o,
            writer.curves_written(),
        );

        // ── Test mode: stop after the first pattern ───────────────────────────
        if cli.test_mode {
            eprintln!("[main] --test-mode: stopping after first pattern.");
            break;
        }
    }

    // ── 7. Finalise the output file ───────────────────────────────────────────
    // Validates total curve count and flushes the OS write buffer.
    writer.finalize().context("Failed to finalise output matrix")?;

    let elapsed = wall_start.elapsed();
    eprintln!(
        "[main] Done. {} curves written to '{}' in {:.1}s",
        n_curves_total,
        cli.output.display(),
        elapsed.as_secs_f64()
    );

    Ok(())
}

// ─── Per-pattern helper ───────────────────────────────────────────────────────

/// Runs stages 1–5 of the pipeline for a single deuteration pattern.
///
/// Returns a [`pepsi::parser::SimulationResult`] containing the Q vector and
/// all I columns for this pattern. The caller is responsible for cleanup
/// (removing `pdb_dir` and `simul_dir`) and for writing the results to the
/// `MatrixWriter`.
///
/// # Why a separate function?
///
/// Isolating the per-pattern work makes error handling clean: `run_pattern`
/// uses `?` freely, and the caller in the main loop can run cleanup
/// unconditionally regardless of whether an error occurred.
///
/// # Parameters
/// - `pattern`     — the current [`DeuterationPattern`].
/// - `mode`        — the AA grouping mode (needed by `apply_pattern`).
/// - `template`    — the immutable reference PDB structure.
/// - `percentages` — ordered D₂O percentage grid.
/// - `pdb_dir`     — where to write `<pct>.pdb` files.
/// - `simul_dir`   — where Pepsi-SANS writes `<pct>.dat` files.
/// - `pepsi_path`  — path to the Pepsi-SANS executable.
/// - `n_threads`   — Rayon thread pool size for `runner::run_all`.
/// - `rng`         — shared RNG for probabilistic labile-H selection.
fn run_pattern(
    pattern:     &binary_vectors::DeuterationPattern,
    mode:        AaMode,
    template:    &pdbrust::PdbStructure,
    percentages: &[u8],
    pdb_dir:     &std::path::Path,
    simul_dir:   &std::path::Path,
    pepsi_path:  &std::path::Path,
    n_threads:   usize,
    rng:         &mut StdRng,
) -> Result<pepsi::parser::SimulationResult> {
    // ── Stage 1: apply non-labile deuteration pattern ─────────────────────────
    let deuterated = pdb::deuteration::apply_pattern(template, *pattern, mode)
        .context("apply_pattern failed")?;

    // ── Stage 2: generate one PDB per %D₂O and write to pdb_dir ─────────────
    std::fs::create_dir_all(pdb_dir)
        .with_context(|| format!("Cannot create '{}'", pdb_dir.display()))?;
    std::fs::create_dir_all(simul_dir)
        .with_context(|| format!("Cannot create '{}'", simul_dir.display()))?;

    for &pct in percentages {
        // Apply probabilistic labile-H exchange.
        let ready = pdb::d2o::apply_percentage(&deuterated, pct, rng)
            .with_context(|| format!("apply_percentage failed at {pct}%"))?;

        // Write the modified PDB to pdbs_<pattern>/<pct>.pdb
        let pdb_path = pdb_dir.join(format!("{pct}.pdb"));
        pdbrust::write_pdb_file(&ready, &pdb_path)
            .with_context(|| format!("Cannot write PDB '{}'", pdb_path.display()))?;
    }

    // ── Stage 3: run Pepsi-SANS in parallel on all PDBs ─────────────────────
    let jobs = pepsi::runner::build_jobs(pdb_dir, simul_dir, percentages);
    pepsi::runner::run_all(&jobs, pepsi_path, n_threads)
        .context("Pepsi-SANS run failed")?;

    // ── Stage 4: parse all .dat output files ─────────────────────────────────
    let sim_result = pepsi::parser::parse_simulation_folder(simul_dir, percentages)
        .context("Parsing Pepsi-SANS output failed")?;

    Ok(sim_result)
}

// ─── Config printer ───────────────────────────────────────────────────────────

/// Prints the resolved configuration to stderr before the run starts.
///
/// Keeps `run()` clean and gives the user a record of what was actually used
/// (useful when the default values apply).
fn print_config(cli: &Cli) {
    eprintln!("=== SANS deuteration pipeline — Phase 1 ===");
    eprintln!("  PDB file   : {}", cli.pdb.display());
    eprintln!("  AA mode    : {:?}", cli.mode);
    eprintln!("  D₂O step   : {}%", cli.d2o_step);
    eprintln!("  Output     : {}", cli.output.display());
    eprintln!("  Pepsi-SANS : {}", cli.pepsi.display());
    eprintln!("  Work dir   : {}", cli.work_dir.display());
    eprintln!(
        "  Threads    : {}",
        if cli.threads == 0 { "auto".to_string() } else { cli.threads.to_string() }
    );
    eprintln!(
        "  RNG seed   : {}",
        cli.seed.map_or_else(|| "random (OS entropy)".to_string(), |s| s.to_string())
    );
    eprintln!(
        "  Test mode  : {}",
        if cli.test_mode { "YES — 1 pattern only" } else { "no" }
    );
    eprintln!(
        "  Keep files : {}",
        if cli.keep_files { "YES — temp dirs kept" } else { "no" }
    );
    eprintln!();
}

// ─── Compatibility notes ──────────────────────────────────────────────────────
//
// The following items must be present in the corresponding modules for this
// file to compile without modification:
//
// binary_vectors::DeuterationPattern
//   - Fields `bits` and `n_bits` are private.
//   - `PatternIter::new(mode)` produces patterns; main.rs never constructs
//     DeuterationPattern directly.
//   - Tests in deuteration.rs call `DeuterationPattern::new_for_test(bits, n_bits)`;
//     add this constructor to binary_vectors.rs:
//
//       #[cfg(test)]
//       impl DeuterationPattern {
//           pub fn new_for_test(bits: u32, n_bits: u32) -> Self {
//               Self { bits, n_bits }
//           }
//       }
//
// pdbrust::write_pdb_file
//   - Signature: `write_pdb_file(structure: &PdbStructure, path: impl AsRef<Path>)
//       -> Result<(), PdbError>`
//   - Re-exported at the crate root in pdbrust 0.7.
//
// pepsi::parser::SimulationResult
//   - Fields `q: Vec<f32>` and `intensities: Vec<Vec<f32>>` must be `pub`.
//
// AaMode implements std::str::FromStr with Err = String.
//   - Clap 4 accepts any type whose `FromStr::Err: Into<Box<dyn Error + Send + Sync>>`.
//   - `String` satisfies this bound via the blanket impl, so no extra work needed.
