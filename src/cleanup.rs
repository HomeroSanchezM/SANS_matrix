//! # `cleanup` — Temporary directory removal
//!
//! After every pattern iteration the pipeline has two temporary directories
//! that are no longer needed:
//!
//! - `pdbs_<pattern>/`  — the per-%D₂O PDB files (input to Pepsi-SANS).
//! - `simul_<pattern>/` — the per-%D₂O `.dat` files (output of Pepsi-SANS).
//!
//! Both were read into the output matrix by [`pepsi::parser`]; keeping them on
//! disk would consume tens of gigabytes for a full 18-bit run.
//!
//! ## Guiding principle (from the spec)
//!
//! > *"Only one binary vector is in memory at a time. Intermediate folders are
//! > deleted after each iteration."*
//!
//! [`remove_pattern_dirs`] is therefore called at the **end** of every pattern
//! loop body in `main.rs`, even on error paths (via the `?` operator chain).
//!
//! ## Error policy
//!
//! Failure to delete a temporary directory is treated as a **non-fatal warning**
//! rather than a hard error: the matrix data has already been written, so
//! aborting the pipeline over a stale temp folder would lose all subsequent
//! curves. The warning is printed to `stderr` so the user can clean up manually.
//!
//! The one exception is [`remove_dir_checked`], which **does** return an error.
//! It is provided for callers (e.g. tests) that need strict error handling.

use std::path::Path;

// ─── Public API ──────────────────────────────────────────────────────────────

/// Deletes the two temporary directories created for `pattern_str`.
///
/// The directories are:
/// - `<work_dir>/pdbs_<pattern_str>/`
/// - `<work_dir>/simul_<pattern_str>/`
///
/// If a directory does not exist the function silently continues — this is
/// normal when, for example, Pepsi-SANS failed before creating `simul_*/`.
///
/// If a directory exists but cannot be removed, a warning is printed to
/// `stderr` and the function continues rather than propagating an error.
/// See the module-level doc for the rationale.
///
/// # Parameters
/// - `work_dir`     — the base directory that contains `pdbs_*/` and `simul_*/`.
/// - `pattern_str`  — the binary string representation of the pattern (e.g.
///                    `"00100000000"`), produced by
///                    [`DeuterationPattern::to_pattern_string`].
///
/// # Example
///
/// ```no_run
/// cleanup::remove_pattern_dirs(Path::new("/tmp/run"), "00100000000");
/// ```
pub fn remove_pattern_dirs(work_dir: &Path, pattern_str: &str) {
    for prefix in ["pdbs", "simul"] {
        let dir = work_dir.join(format!("{prefix}_{pattern_str}"));

        // Nothing to do if the directory was never created.
        if !dir.exists() {
            continue;
        }

        if let Err(e) = std::fs::remove_dir_all(&dir) {
            // Non-fatal: warn and move on.  The matrix is already written.
            eprintln!(
                "[cleanup] WARNING: could not remove '{}': {e}",
                dir.display()
            );
        } else {
            // Verbose confirmation at debug level helps trace memory usage.
            eprintln!("[cleanup] Removed '{}'", dir.display());
        }
    }
}

/// Removes a single directory and all its contents, returning an error on
/// failure.
///
/// Unlike [`remove_pattern_dirs`], this function propagates the I/O error to
/// the caller. Use it when the caller must guarantee the directory is gone
/// (e.g. before creating it again, or in tests that verify cleanup).
///
/// Silently succeeds if the directory does not exist (idempotent).
///
/// # Errors
///
/// Returns an error if the directory exists but cannot be removed.
#[allow(dead_code)]
pub fn remove_dir_checked(dir: &Path) -> std::io::Result<()> {
    match std::fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()), // already gone
        Err(e) => Err(e),
    }
}

/// Returns the path of the PDB directory for `pattern_str`.
///
/// Convenience so callers never have to hand-format the directory name.
/// The returned path is `<work_dir>/pdbs_<pattern_str>`.
#[inline]
pub fn pdb_dir(work_dir: &Path, pattern_str: &str) -> std::path::PathBuf {
    work_dir.join(format!("pdbs_{pattern_str}"))
}

/// Returns the path of the simulation output directory for `pattern_str`.
///
/// The returned path is `<work_dir>/simul_<pattern_str>`.
#[inline]
pub fn simul_dir(work_dir: &Path, pattern_str: &str) -> std::path::PathBuf {
    work_dir.join(format!("simul_{pattern_str}"))
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── remove_pattern_dirs ───────────────────────────────────────────────────

    #[test]
    fn removes_both_dirs() {
        let root = TempDir::new().unwrap();
        let pattern = "00100000000";

        let pdbs  = pdb_dir(root.path(), pattern);
        let simul = simul_dir(root.path(), pattern);

        std::fs::create_dir_all(&pdbs).unwrap();
        std::fs::create_dir_all(&simul).unwrap();
        // Place a dummy file inside each to confirm recursive removal.
        std::fs::write(pdbs.join("0.pdb"), b"").unwrap();
        std::fs::write(simul.join("0.dat"), b"").unwrap();

        remove_pattern_dirs(root.path(), pattern);

        assert!(!pdbs.exists(),  "pdbs dir should be gone");
        assert!(!simul.exists(), "simul dir should be gone");
    }

    #[test]
    fn nonexistent_dirs_are_silently_skipped() {
        let root = TempDir::new().unwrap();
        // Neither directory exists — should not panic.
        remove_pattern_dirs(root.path(), "11111111111");
    }

    #[test]
    fn partial_cleanup_when_only_pdbs_exists() {
        let root = TempDir::new().unwrap();
        let pattern = "00000000001";

        let pdbs = pdb_dir(root.path(), pattern);
        std::fs::create_dir_all(&pdbs).unwrap();
        // simul dir does not exist

        remove_pattern_dirs(root.path(), pattern);

        assert!(!pdbs.exists(), "pdbs dir should be gone");
    }

    // ── remove_dir_checked ────────────────────────────────────────────────────

    #[test]
    fn checked_removes_existing_dir() {
        let root = TempDir::new().unwrap();
        let dir = root.path().join("to_remove");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("file.txt"), b"data").unwrap();

        remove_dir_checked(&dir).unwrap();
        assert!(!dir.exists());
    }

    #[test]
    fn checked_is_ok_when_dir_does_not_exist() {
        let dir = std::path::PathBuf::from("/tmp/nonexistent_sans_dir_xyz123");
        assert!(remove_dir_checked(&dir).is_ok());
    }

    // ── path helpers ──────────────────────────────────────────────────────────

    #[test]
    fn pdb_dir_path_is_correct() {
        let p = pdb_dir(Path::new("/work"), "00100");
        assert_eq!(p, std::path::PathBuf::from("/work/pdbs_00100"));
    }

    #[test]
    fn simul_dir_path_is_correct() {
        let p = simul_dir(Path::new("/work"), "00100");
        assert_eq!(p, std::path::PathBuf::from("/work/simul_00100"));
    }
}
