//! # `binary_vectors` — Deuteration pattern generation
//!
//! This module provides:
//!
//! - [`DeuterationPattern`] — a newtype wrapping a `u32` that encodes one
//!   assignment of protonated/deuterated state across all amino-acid groups.
//! - [`PatternIter`] — a lazy iterator that yields every pattern from
//!   `0` (all protonated) to `2^n_bits − 1` (all deuterated) without
//!   allocating a `Vec` upfront.
//!
//! ## Bit convention
//!
//! Bit `i` corresponds to the group at index `i` in the [`AaMode::groups()`]
//! slice. Bit `0` is the least-significant bit.
//!
//! ```text
//! Reduced mode, bit layout:
//!   bit 0 → ARG
//!   bit 1 → GLU/GLN
//!   bit 2 → HIS
//!   ...
//!   bit 10 → VAL
//!
//! Pattern 0b00000000100  (decimal 4)
//!   → only bit 2 is set  → only HIS is deuterated
//!   → string: "00100000000"  (MSB … LSB, matching the spec)
//! ```
//!
//! ## String representation
//!
//! [`DeuterationPattern::to_pattern_string`] formats the pattern with the
//! **most-significant group first** (i.e. index `n_bits-1` on the left),
//! matching the human-readable examples in the project specification and
//! used as directory-name suffixes (`pdbs_<pattern>/`, `simul_<pattern>/`).
//!
//! ## Design notes
//! - `DeuterationPattern` is `Copy` — it is just a `u32` plus a bit count.
//! - `PatternIter` holds no heap state; it is a `u64` counter.
//! - Both `n_bits` and the pattern value fit comfortably in 32 bits because
//!   the maximum bit-width used by the project is 18.

use crate::amino_acids::{AaGroup, AaMode};

// ─── DeuterationPattern ──────────────────────────────────────────────────────

/// One assignment of protonated/deuterated state for every amino-acid group.
///
/// Internally stored as a `u32` bitmask. Bit `i` is `1` when the group at
/// index `i` in the current [`AaMode`] is **deuterated**.
///
/// Use [`PatternIter`] to iterate over all valid patterns for a given mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeuterationPattern {
    /// The bitmask. Only the lowest `n_bits` bits are meaningful.
    bits: u32,
    /// Number of amino-acid groups (= number of significant bits).
    n_bits: u32,
}

impl DeuterationPattern {
    /// Returns `true` if the group at `group_idx` is **deuterated** in this
    /// pattern, `false` if it is protonated.
    ///
    /// # Panics
    /// Panics in debug builds if `group_idx >= n_bits`.
    #[inline]
    pub fn is_deuterated(self, group_idx: usize) -> bool {
        debug_assert!(
            (group_idx as u32) < self.n_bits,
            "group_idx {group_idx} out of range for {}-bit pattern",
            self.n_bits
        );
        (self.bits >> group_idx) & 1 == 1
    }

    /// Returns the number of amino-acid groups encoded in this pattern.
    #[allow(dead_code)]
    #[inline]
    pub fn n_bits(self) -> u32 {
        self.n_bits
    }

    /// Returns the raw bitmask value (useful for arithmetic or storage).
    #[allow(dead_code)]
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.bits
    }

    /// Formats the pattern as a binary string with the **highest-index group
    /// on the left**, matching the project specification convention.
    ///
    /// # Example
    /// ```
    /// // Reduced mode (11 bits), only HIS (index 2) deuterated → bits = 0b100 = 4
    /// // Expected string: "00000000100"  (index 10 … index 0)
    /// ```
    pub fn to_pattern_string(self) -> String {
        // Iterate from the most-significant bit down to bit 0.
        (0..self.n_bits)
            .rev()
            .map(|i| if (self.bits >> i) & 1 == 1 { '1' } else { '0' })
            .collect()
    }

    /// Returns an iterator over `(group_index, &AaGroup, is_deuterated)` for
    /// every group in `mode`, in ascending index order.
    ///
    /// Useful when building the modified PDB: iterate once, apply changes only
    /// to groups where `is_deuterated` is `true`.
    #[allow(dead_code)]
    pub fn iter_groups<'a>(
        self,
        mode: AaMode,
    ) -> impl Iterator<Item = (usize, &'a AaGroup, bool)> + 'a {
        mode.groups()
            .iter()
            .enumerate()
            .map(move |(i, group)| (i, group, self.is_deuterated(i)))
    }
}

/// Displays the pattern in the same format as [`to_pattern_string`](DeuterationPattern::to_pattern_string).
impl std::fmt::Display for DeuterationPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_pattern_string())
    }
}

// ─── PatternIter ─────────────────────────────────────────────────────────────

/// A lazy iterator that yields every [`DeuterationPattern`] for a given
/// [`AaMode`], from `0` (all protonated) to `2^n_bits − 1` (all deuterated).
///
/// No allocation is performed; the iterator is a single `u64` counter.
///
/// # Example
/// ```
/// use crate::amino_acids::AaMode;
/// use crate::binary_vectors::PatternIter;
///
/// let total = AaMode::Reduced.n_patterns();   // 2048
/// let iter  = PatternIter::new(AaMode::Reduced);
/// assert_eq!(iter.count(), total as usize);
/// ```
pub struct PatternIter {
    /// Current counter value (cast to u32 for each yielded pattern).
    current: u64,
    /// Exclusive upper bound: 2^n_bits.
    total: u64,
    /// Number of significant bits in each yielded pattern.
    n_bits: u32,
}

impl PatternIter {
    /// Creates an iterator over all patterns for `mode`.
    pub fn new(mode: AaMode) -> Self {
        Self {
            current: 0,
            total: mode.n_patterns(),
            n_bits: mode.n_bits(),
        }
    }
}

impl Iterator for PatternIter {
    type Item = DeuterationPattern;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total {
            return None;
        }
        // Safety: current < 2^n_bits ≤ 2^18, fits in u32.
        let pattern = DeuterationPattern {
            bits: self.current as u32,
            n_bits: self.n_bits,
        };
        self.current += 1;
        Some(pattern)
    }

    /// Provides an exact size hint so callers can pre-allocate if needed.
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.total - self.current) as usize;
        (remaining, Some(remaining))
    }
}

// Marks the iterator as having a known exact length (enables `.count()` etc.).
impl ExactSizeIterator for PatternIter {}

// ─── Test-only constructor ────────────────────────────────────────────────────

#[cfg(test)]
impl DeuterationPattern {
    /// Constructs a [`DeuterationPattern`] directly from its raw parts.
    ///
    /// **Only available in test code.** Production code always obtains patterns
    /// from [`PatternIter`], which guarantees that `bits < 2^n_bits`.
    /// This constructor skips that invariant so individual unit tests in other
    /// modules (e.g. `pdb::deuteration`) can build specific patterns without
    /// going through the iterator.
    ///
    /// # Parameters
    /// - `bits`   — the bitmask; only the lowest `n_bits` bits are meaningful.
    /// - `n_bits` — number of amino-acid groups encoded (e.g. 11 for `Reduced`).
    pub fn new_for_test(bits: u32, n_bits: u32) -> Self {
        Self { bits, n_bits }
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amino_acids::AaMode;

    // ── DeuterationPattern ────────────────────────────────────────────────

    #[test]
    fn all_protonated_is_zero() {
        let p = DeuterationPattern { bits: 0, n_bits: 11 };
        for i in 0..11 {
            assert!(!p.is_deuterated(i));
        }
        assert_eq!(p.to_pattern_string(), "00000000000");
    }

    #[test]
    fn all_deuterated_string() {
        // Reduced: 11 bits, all ones → "11111111111"
        let p = DeuterationPattern { bits: (1 << 11) - 1, n_bits: 11 };
        assert_eq!(p.to_pattern_string(), "11111111111");
    }

    #[test]
    fn single_bit_set_his_index_2_reduced() {
        // HIS is index 2 in the reduced list → bit 2 → bits = 0b00000000100 = 4
        let p = DeuterationPattern { bits: 1 << 2, n_bits: 11 };
        assert!(p.is_deuterated(2));
        assert!(!p.is_deuterated(0));
        assert!(!p.is_deuterated(10));
        // String: MSB (index 10) … LSB (index 0) → "00000000100"
        assert_eq!(p.to_pattern_string(), "00000000100");
    }

    #[test]
    fn display_matches_pattern_string() {
        let p = DeuterationPattern { bits: 0b10100, n_bits: 11 };
        assert_eq!(format!("{p}"), p.to_pattern_string());
    }

    // ── PatternIter ───────────────────────────────────────────────────────

    #[test]
    fn reduced_yields_2048_patterns() {
        let count = PatternIter::new(AaMode::Reduced).count();
        assert_eq!(count, 2_048);
    }

    #[test]
    fn complete_yields_262144_patterns() {
        let count = PatternIter::new(AaMode::Complete).count();
        assert_eq!(count, 262_144);
    }

    #[test]
    fn first_pattern_is_all_protonated() {
        let first = PatternIter::new(AaMode::Reduced).next().unwrap();
        assert_eq!(first.as_u32(), 0);
        assert_eq!(first.to_pattern_string(), "00000000000");
    }

    #[test]
    fn last_pattern_is_all_deuterated() {
        let last = PatternIter::new(AaMode::Reduced).last().unwrap();
        assert_eq!(last.as_u32(), (1 << 11) - 1);
        assert_eq!(last.to_pattern_string(), "11111111111");
    }

    #[test]
    fn size_hint_is_exact() {
        let mut iter = PatternIter::new(AaMode::Reduced);
        assert_eq!(iter.size_hint(), (2_048, Some(2_048)));
        iter.next();
        assert_eq!(iter.size_hint(), (2_047, Some(2_047)));
    }

    #[test]
    fn patterns_are_unique() {
        // Collect all patterns and verify no duplicates via a HashSet on u32.
        use std::collections::HashSet;
        let bits: HashSet<u32> = PatternIter::new(AaMode::Reduced)
            .map(|p| p.as_u32())
            .collect();
        assert_eq!(bits.len(), 2_048);
    }

    #[test]
    fn iter_groups_counts_deuterated_correctly() {
        // Pattern with exactly 3 bits set.
        let p = DeuterationPattern { bits: 0b111, n_bits: 11 };
        let deuterated_count = p
            .iter_groups(AaMode::Reduced)
            .filter(|(_, _, d)| *d)
            .count();
        assert_eq!(deuterated_count, 3);
    }
}
