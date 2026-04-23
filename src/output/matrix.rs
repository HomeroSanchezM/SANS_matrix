//! # `output::matrix` — Incremental `f32` binary matrix writer
//!
//! [`MatrixWriter`] writes the output matrix **one column at a time** to a
//! binary file, without ever holding more than one curve in memory.
//!
//! ## State machine
//!
//! The writer enforces a strict call order through an internal state enum:
//!
//! ```text
//! ┌──────────┐  write_q()  ┌─────────┐  write_curve() × N  ┌──────────┐
//! │ NeedQ    │────────────▶│ Writing │────────────────────▶│ Finished │
//! └──────────┘             └─────────┘                      └──────────┘
//!                                │
//!                         finalize() at any point
//!                         (validates total curve count)
//! ```
//!
//! - Calling [`write_curve`](MatrixWriter::write_curve) before
//!   [`write_q`](MatrixWriter::write_q) returns an error.
//! - Calling [`write_q`](MatrixWriter::write_q) twice returns an error.
//! - [`finalize`](MatrixWriter::finalize) validates that exactly
//!   `n_curves` curves were written and flushes the OS buffer.
//!
//! ## Memory usage
//!
//! `MatrixWriter` itself holds only a `BufWriter<File>` (an 8 KB write buffer)
//! plus a few counters — **O(1)** regardless of matrix size.
//!
//! ## Number of Q points
//!
//! The expected number of Q points per curve is set at construction time
//! (`n_q_points`, normally [`pepsi::parser::EXPECTED_POINTS`] = 50). Any
//! call to [`write_q`](MatrixWriter::write_q) or
//! [`write_curve`](MatrixWriter::write_curve) with a slice of a different
//! length returns an error immediately.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};

// ─── Writer state ─────────────────────────────────────────────────────────────

/// Internal state of the [`MatrixWriter`] state machine.
#[derive(Debug, PartialEq, Eq)]
enum WriterState {
    /// Q column has not been written yet. No curves may be written.
    NeedQ,
    /// Q column written; curves are being appended one by one.
    Writing,
    /// `finalize()` has been called; no further writes are allowed.
    Finished,
}

// ─── MatrixWriter ─────────────────────────────────────────────────────────────

/// Writes the SANS output matrix incrementally to a binary `f32` file.
///
/// The file is opened for writing when [`MatrixWriter::new`] is called and
/// closed (flushed) when [`MatrixWriter::finalize`] is called.
///
/// ## Example
///
/// ```no_run
/// use output::MatrixWriter;
///
/// let total_curves = 2048u64 * 21; // reduced mode, 21 D₂O steps
/// let mut writer = MatrixWriter::new("output.bin", total_curves, 50)?;
///
/// writer.write_q(&q_vec)?;
///
/// for pattern in pattern_iter {
///     for pct in &d2o_percentages {
///         let i_vec = /* … compute one curve … */;
///         writer.write_curve(&i_vec)?;
///     }
/// }
///
/// writer.finalize()?;
/// ```
pub struct MatrixWriter {
    /// Buffered file writer. The OS buffer is flushed on `finalize()`.
    writer: BufWriter<File>,

    /// Expected number of `f32` values per column (Q points).
    n_q_points: usize,

    /// Total number of I-curves expected (not counting the Q column).
    n_curves_expected: u64,

    /// Number of I-curves actually written so far.
    n_curves_written: u64,

    /// Current state of the state machine.
    state: WriterState,
}

impl MatrixWriter {
    /// Creates (or truncates) the output file and prepares the writer.
    ///
    /// The file is not pre-allocated; it grows incrementally as columns are
    /// appended. This avoids allocating gigabytes of disk space upfront on
    /// systems that do not support sparse files.
    ///
    /// # Parameters
    /// - `path`              — path to the output binary file.
    /// - `n_curves_expected` — total number of I-curves that will be written
    ///                         (= N_patterns × N_d2o). Used only for the final
    ///                         count validation in [`finalize`](Self::finalize).
    /// - `n_q_points`        — number of Q values per curve (normally 50).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or opened.
    pub fn new(path: &Path, n_curves_expected: u64, n_q_points: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true) // overwrite any previous run
            .open(path)
            .with_context(|| format!("Cannot create output file '{}'", path.display()))?;

        // 8 KB write buffer — amortises syscall overhead for the 200-byte
        // column writes without wasting memory.
        let writer = BufWriter::new(file);

        Ok(Self {
            writer,
            n_q_points,
            n_curves_expected,
            n_curves_written: 0,
            state: WriterState::NeedQ,
        })
    }

    /// Writes the shared Q vector as column 0 of the matrix.
    ///
    /// This method **must** be called exactly once, before any call to
    /// [`write_curve`](Self::write_curve).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Called more than once.
    /// - `q.len() != n_q_points`.
    /// - The underlying write fails.
    pub fn write_q(&mut self, q: &[f32]) -> Result<()> {
        if self.state != WriterState::NeedQ {
            bail!("write_q() called more than once or after finalize()");
        }
        self.write_column(q).context("Failed to write Q column")?;
        self.state = WriterState::Writing;
        Ok(())
    }

    /// Appends one scattering intensity curve as the next column.
    ///
    /// Curves must be written in the canonical order: for each pattern (outer
    /// loop), for each D₂O percentage in ascending order (inner loop). This
    /// order is encoded in the column index and determines how Phase 2 maps a
    /// column back to its `(pattern, pct)` condition.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Called before [`write_q`](Self::write_q).
    /// - `finalize()` has already been called.
    /// - `i.len() != n_q_points`.
    /// - More curves are written than `n_curves_expected`.
    /// - The underlying write fails.
    pub fn write_curve(&mut self, i: &[f32]) -> Result<()> {
        match self.state {
            WriterState::NeedQ => bail!(
                "write_curve() called before write_q() — Q column must be written first"
            ),
            WriterState::Finished => bail!(
                "write_curve() called after finalize() — writer is closed"
            ),
            WriterState::Writing => {}
        }

        if self.n_curves_written >= self.n_curves_expected {
            bail!(
                "Attempted to write curve {} but only {} were expected",
                self.n_curves_written + 1,
                self.n_curves_expected
            );
        }

        self.write_column(i).with_context(|| {
            format!(
                "Failed to write curve {} of {}",
                self.n_curves_written + 1,
                self.n_curves_expected
            )
        })?;

        self.n_curves_written += 1;
        Ok(())
    }

    /// Returns the number of I-curves written so far (not counting the Q column).
    #[inline]
    pub fn curves_written(&self) -> u64 {
        self.n_curves_written
    }

    /// Flushes the write buffer and validates the total curve count.
    ///
    /// After this call the file is complete and the writer is closed.
    /// Calling any write method after `finalize()` returns an error.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of written curves does not match `n_curves_expected`.
    /// - The OS buffer flush fails.
    /// - Called before [`write_q`](Self::write_q) (Q column missing).
    pub fn finalize(mut self) -> Result<()> {
        if self.state == WriterState::NeedQ {
            bail!("finalize() called without writing the Q column first");
        }

        // Validate total curve count before flushing — fail loudly if the
        // pipeline dropped any pattern.
        if self.n_curves_written != self.n_curves_expected {
            bail!(
                "Matrix is incomplete: wrote {} curves but expected {}",
                self.n_curves_written,
                self.n_curves_expected
            );
        }

        // Flush the BufWriter so all bytes reach the OS (and eventually disk).
        self.writer
            .flush()
            .context("Failed to flush output file")?;

        self.state = WriterState::Finished;
        Ok(())
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Serialises `values` as little-endian `f32` bytes and appends them to
    /// the file.
    ///
    /// Each `f32` becomes exactly 4 bytes → one call writes `n_q_points × 4`
    /// bytes (200 bytes for the standard 50-point grid).
    ///
    /// # Errors
    ///
    /// Returns an error if `values.len() != n_q_points` or the write fails.
    fn write_column(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.n_q_points {
            bail!(
                "Column has {} values but {} were expected",
                values.len(),
                self.n_q_points
            );
        }

        // Convert each f32 to 4 little-endian bytes and write them.
        // We write value-by-value rather than transmuting the whole slice
        // to avoid undefined behaviour on big-endian targets and to keep
        // the byte order explicit and testable.
        for &val in values {
            self.writer
                .write_all(&val.to_le_bytes())
                .context("I/O error writing f32 value")?;
        }

        Ok(())
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use tempfile::NamedTempFile;

    const N_Q: usize = 4; // small N for tests; real value is 50

    /// Creates a MatrixWriter backed by a temp file.
    fn make_writer(n_curves: u64) -> (MatrixWriter, NamedTempFile) {
        let f = NamedTempFile::new().unwrap();
        let w = MatrixWriter::new(f.path(), n_curves, N_Q).unwrap();
        (w, f)
    }

    /// Reads all bytes from the temp file and interprets them as f32.
    fn read_f32s(f: &NamedTempFile) -> Vec<f32> {
        let mut buf = Vec::new();
        File::open(f.path()).unwrap().read_to_end(&mut buf).unwrap();
        buf.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    // ── Happy path ───────────────────────────────────────────────────────────

    #[test]
    fn writes_q_and_two_curves_correctly() {
        let (mut w, f) = make_writer(2);

        let q = vec![0.0f32, 0.1, 0.2, 0.3];
        let i0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let i1 = vec![5.0f32, 6.0, 7.0, 8.0];

        w.write_q(&q).unwrap();
        w.write_curve(&i0).unwrap();
        w.write_curve(&i1).unwrap();
        w.finalize().unwrap();

        // Expected flat layout: [q[0..4], i0[0..4], i1[0..4]]
        let data = read_f32s(&f);
        assert_eq!(data, vec![0.0, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn total_file_size_is_correct() {
        let n_curves = 5u64;
        let (mut w, f) = make_writer(n_curves);

        w.write_q(&vec![0.0f32; N_Q]).unwrap();
        for _ in 0..n_curves {
            w.write_curve(&vec![1.0f32; N_Q]).unwrap();
        }
        w.finalize().unwrap();

        let expected_bytes = (1 + n_curves as usize) * N_Q * 4;
        let actual_bytes = f.as_file().metadata().unwrap().len() as usize;
        assert_eq!(actual_bytes, expected_bytes);
    }

    #[test]
    fn curves_written_counter_increments() {
        let (mut w, _f) = make_writer(3);
        w.write_q(&vec![0.0f32; N_Q]).unwrap();
        assert_eq!(w.curves_written(), 0);
        w.write_curve(&vec![0.0f32; N_Q]).unwrap();
        assert_eq!(w.curves_written(), 1);
        w.write_curve(&vec![0.0f32; N_Q]).unwrap();
        assert_eq!(w.curves_written(), 2);
    }

    // ── State machine violations ─────────────────────────────────────────────

    #[test]
    fn write_curve_before_write_q_is_error() {
        let (mut w, _f) = make_writer(1);
        assert!(w.write_curve(&vec![0.0f32; N_Q]).is_err());
    }

    #[test]
    fn write_q_twice_is_error() {
        let (mut w, _f) = make_writer(1);
        w.write_q(&vec![0.0f32; N_Q]).unwrap();
        assert!(w.write_q(&vec![0.0f32; N_Q]).is_err());
    }

    #[test]
    fn finalize_without_write_q_is_error() {
        let (w, _f) = make_writer(0);
        assert!(w.finalize().is_err());
    }

    #[test]
    fn finalize_with_missing_curves_is_error() {
        let (mut w, _f) = make_writer(3); // expect 3 curves
        w.write_q(&vec![0.0f32; N_Q]).unwrap();
        w.write_curve(&vec![0.0f32; N_Q]).unwrap(); // only 1 written
        assert!(w.finalize().is_err());
    }

    #[test]
    fn writing_more_curves_than_expected_is_error() {
        let (mut w, _f) = make_writer(1); // only 1 expected
        w.write_q(&vec![0.0f32; N_Q]).unwrap();
        w.write_curve(&vec![0.0f32; N_Q]).unwrap();
        assert!(w.write_curve(&vec![0.0f32; N_Q]).is_err()); // 2nd → error
    }

    // ── Wrong column length ──────────────────────────────────────────────────

    #[test]
    fn write_q_wrong_length_is_error() {
        let (mut w, _f) = make_writer(1);
        assert!(w.write_q(&vec![0.0f32; N_Q + 1]).is_err());
    }

    #[test]
    fn write_curve_wrong_length_is_error() {
        let (mut w, _f) = make_writer(1);
        w.write_q(&vec![0.0f32; N_Q]).unwrap();
        assert!(w.write_curve(&vec![0.0f32; N_Q - 1]).is_err());
    }

    // ── Byte-order verification ───────────────────────────────────────────────

    #[test]
    fn values_are_stored_as_little_endian() {
        // Write a single known f32 and check the raw bytes.
        let ( w, f) = make_writer(0);
        // We can't call finalize() with 0 curves without writing Q, but we can
        // test write_column indirectly via write_q on a 1-element "grid".
        let mut w1 = MatrixWriter::new(f.path(), 0, 1).unwrap();
        w1.write_q(&[1.0f32]).unwrap();
        // Drop writer to flush (simpler than calling finalize here).
        drop(w1);

        let mut raw = [0u8; 4];
        File::open(f.path())
            .unwrap()
            .read_exact(&mut raw)
            .unwrap();

        // 1.0_f32 in little-endian is [0x00, 0x00, 0x80, 0x3F].
        assert_eq!(raw, 1.0f32.to_le_bytes());

        drop(w); // silence unused warning
    }

    // ── Zero-curve edge case ─────────────────────────────────────────────────

    #[test]
    fn zero_curves_writes_only_q() {
        let (mut w, f) = make_writer(0);
        w.write_q(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        w.finalize().unwrap();

        let data = read_f32s(&f);
        assert_eq!(data, vec![1.0f32, 2.0, 3.0, 4.0]);
    }
}
