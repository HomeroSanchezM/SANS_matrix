//! # `output` — Incremental binary matrix writer
//!
//! This module writes the final output of the SANS pipeline: a binary `f32`
//! matrix where every column is one scattering curve (one combination of
//! deuteration pattern and D₂O percentage).
//!
//! ## Logical matrix layout
//!
//! ```text
//!            col 0    col 1          col 2          …   col K
//!           ┌──────┬──────────────┬──────────────┬───┬──────────────┐
//! row  0    │ q[0] │ I(pat0,d2o0) │ I(pat0,d2o1) │ … │ I(patN,d2oM) │
//! row  1    │ q[1] │     …        │     …        │   │      …       │
//!  …        │  …   │     …        │     …        │   │      …       │
//! row 49    │q[49] │     …        │     …        │   │      …       │
//!           └──────┴──────────────┴──────────────┴───┴──────────────┘
//! ```
//!
//! - **Column 0** — the shared Q vector (50 scattering angles, in Å⁻¹).
//! - **Columns 1 … K** — one I(q) curve per `(pattern, %D₂O)` pair, in the
//!   order they are generated (pattern index increases in the outer loop;
//!   D₂O percentage increases in the inner loop).
//!
//! ## On-disk storage format
//!
//! Columns are stored **sequentially** (column-major order): all 50 `f32`
//! values of column 0, then all 50 of column 1, etc. Each value is written
//! in **little-endian** byte order (native on x86/ARM).
//!
//! ```text
//! File bytes:
//!   [q[0] q[1] … q[49]]  [I_c1[0] … I_c1[49]]  [I_c2[0] … I_c2[49]]  …
//!    \___ column 0 ___/    \_____ column 1 _____/  \_____ column 2 _____/
//! ```
//!
//! Total size: `(1 + N_curves) × 50 × 4` bytes.
//!
//! ## Why column-major and not row-major?
//!
//! The pipeline generates up to **5.5 million curves** one at a time. Writing
//! them in row-major order (interleaved Q-point by Q-point) would require
//! random-access I/O with a stride of `N_curves × 4` bytes — tens of
//! megabytes per jump — causing millions of cache misses and thrashing the OS
//! page cache.
//!
//! Sequential column-major writes are cache-optimal: every `write_curve` call
//! is a contiguous 200-byte append. For a 1 GB output file, the difference in
//! wall time is roughly 10–100×.
//!
//! ## Reading in Python (Phase 2)
//!
//! ```python
//! import numpy as np
//!
//! # Each row of `raw` is one full curve (50 Q-points).
//! raw = np.fromfile("output.bin", dtype=np.float32).reshape(-1, 50)
//!
//! Q     = raw[0]     # shape (50,)        — shared Q vector
//! I_all = raw[1:].T  # shape (50, N_curves) — one column per curve
//! ```
//!
//! Or equivalently, using Fortran order to get the exact (50, N_cols) shape:
//!
//! ```python
//! matrix = np.fromfile("output.bin", dtype=np.float32).reshape(50, -1, order='F')
//! Q     = matrix[:, 0]
//! I_all = matrix[:, 1:]   # shape (50, N_curves)
//! ```

pub mod matrix;

// Re-export so callers can write `output::MatrixWriter` instead of
// `output::matrix::MatrixWriter`.
pub use matrix::MatrixWriter;
