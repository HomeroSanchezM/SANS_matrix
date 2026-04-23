//! # `amino_acids` — Amino-acid group definitions
//!
//! This module defines the two operating modes of the deuteration pipeline:
//! - [`AaMode::Complete`]: 18 groups, yielding 2^18 = 262 144 binary patterns.
//! - [`AaMode::Reduced`]: 11 groups, yielding 2^11 = 2 048 binary patterns.
//!
//! Each *group* is a named bucket of one or more amino-acid residue types that
//! share the same deuteration state (either all protonated or all deuterated).
//! The three-letter residue names stored here must match the `resName` field
//! written by Pepsi-SANS / pdbrust exactly (e.g. `"ALA"`, `"GLU"`).
//!
//! ## Design notes
//! - All data is `'static`; no heap allocation is needed at runtime.
//! - [`AaMode`] implements [`std::str::FromStr`] so it can be parsed directly
//!   from a CLI `--mode` argument without extra glue code.
//! - [`AaGroup`] is intentionally a plain struct with public fields; it is
//!   read-only after construction and never mutated.

// ─── Public types ────────────────────────────────────────────────────────────

/// One amino-acid group: a label and the residue names it covers.
///
/// The `residues` slice always contains at least one entry. When a group
/// covers several residues (e.g. `"ASN/ASP"`) all of them share the same
/// deuteration bit.
#[derive(Debug, Clone, Copy)]
pub struct AaGroup {
    /// Human-readable label used in log messages and folder names.
    #[allow(dead_code)]
    pub label: &'static str,
    /// Three-letter residue codes that belong to this group.
    pub residues: &'static [&'static str],
}

/// Selects which amino-acid grouping scheme to use.
///
/// The `u8` discriminants are **not** part of the public API; use the named
/// variants everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AaMode {
    /// 18 groups → 2^18 = 262 144 deuteration patterns.
    Complete,
    /// 11 groups → 2^11 = 2 048 deuteration patterns.
    Reduced,
}

impl AaMode {
    /// Returns the ordered slice of [`AaGroup`]s for this mode.
    ///
    /// The index of each group in the returned slice **is** the bit position
    /// it occupies in a [`DeuterationPattern`](crate::binary_vectors::DeuterationPattern).
    /// Index 0 is the least-significant bit.
    #[inline]
    pub fn groups(self) -> &'static [AaGroup] {
        match self {
            AaMode::Complete => COMPLETE_GROUPS,
            AaMode::Reduced => REDUCED_GROUPS,
        }
    }

    /// Number of bits (= number of groups) for this mode.
    #[inline]
    pub fn n_bits(self) -> u32 {
        self.groups().len() as u32
    }

    /// Total number of distinct deuteration patterns: 2^[`n_bits`](Self::n_bits).
    #[inline]
    pub fn n_patterns(self) -> u64 {
        1u64 << self.n_bits()
    }
}

// ─── FromStr — parses "complete" / "reduced" from CLI ────────────────────────

impl std::str::FromStr for AaMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "complete" => Ok(AaMode::Complete),
            "reduced" => Ok(AaMode::Reduced),
            other => Err(format!(
                "unknown mode `{other}`; expected `complete` or `reduced`"
            )),
        }
    }
}

// ─── Static group tables ─────────────────────────────────────────────────────

/// **Complete** grouping: 18 groups, one per amino-acid type (or merged pair).
///
/// Order matches the specification table: index 0 = ALA, … index 17 = VAL.
/// This order must not change because it determines the bit layout of every
/// stored [`DeuterationPattern`](crate::binary_vectors::DeuterationPattern).
static COMPLETE_GROUPS: &[AaGroup] = &[
    AaGroup { label: "ALA",     residues: &["ALA"] },
    AaGroup { label: "ARG",     residues: &["ARG"] },
    AaGroup { label: "ASN/ASP", residues: &["ASN", "ASP"] },
    AaGroup { label: "CYS",     residues: &["CYS"] },
    AaGroup { label: "GLU/GLN", residues: &["GLU", "GLN"] },
    AaGroup { label: "GLY",     residues: &["GLY"] },
    AaGroup { label: "HIS",     residues: &["HIS"] },
    AaGroup { label: "ILE",     residues: &["ILE"] },
    AaGroup { label: "LEU",     residues: &["LEU"] },
    AaGroup { label: "LYS",     residues: &["LYS"] },
    AaGroup { label: "MET",     residues: &["MET"] },
    AaGroup { label: "PHE",     residues: &["PHE"] },
    AaGroup { label: "PRO",     residues: &["PRO"] },
    AaGroup { label: "SER",     residues: &["SER"] },
    AaGroup { label: "THR",     residues: &["THR"] },
    AaGroup { label: "TRP",     residues: &["TRP"] },
    AaGroup { label: "TYR",     residues: &["TYR"] },
    AaGroup { label: "VAL",     residues: &["VAL"] },
];

/// **Reduced** grouping: 11 groups containing only residues with significant
/// SANS contrast variation — the ones most likely to produce distinguishable
/// scattering curves.
///
/// Order: index 0 = ARG, … index 10 = VAL. Same bit-layout caveat as above.
static REDUCED_GROUPS: &[AaGroup] = &[
    AaGroup { label: "ARG",     residues: &["ARG"] },
    AaGroup { label: "GLU/GLN", residues: &["GLU", "GLN"] },
    AaGroup { label: "HIS",     residues: &["HIS"] },
    AaGroup { label: "ILE",     residues: &["ILE"] },
    AaGroup { label: "LEU",     residues: &["LEU"] },
    AaGroup { label: "LYS",     residues: &["LYS"] },
    AaGroup { label: "PHE",     residues: &["PHE"] },
    AaGroup { label: "PRO",     residues: &["PRO"] },
    AaGroup { label: "THR",     residues: &["THR"] },
    AaGroup { label: "TYR",     residues: &["TYR"] },
    AaGroup { label: "VAL",     residues: &["VAL"] },
];

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn complete_has_18_groups() {
        assert_eq!(AaMode::Complete.n_bits(), 18);
    }

    #[test]
    fn reduced_has_11_groups() {
        assert_eq!(AaMode::Reduced.n_bits(), 11);
    }

    #[test]
    fn n_patterns_correct() {
        assert_eq!(AaMode::Complete.n_patterns(), 262_144);
        assert_eq!(AaMode::Reduced.n_patterns(), 2_048);
    }

    #[test]
    fn fromstr_case_insensitive() {
        assert_eq!(AaMode::from_str("Complete").unwrap(), AaMode::Complete);
        assert_eq!(AaMode::from_str("REDUCED").unwrap(), AaMode::Reduced);
    }

    #[test]
    fn fromstr_rejects_unknown() {
        assert!(AaMode::from_str("full").is_err());
    }

    #[test]
    fn no_group_has_empty_residue_list() {
        for mode in [AaMode::Complete, AaMode::Reduced] {
            for g in mode.groups() {
                assert!(!g.residues.is_empty(), "group `{}` has no residues", g.label);
            }
        }
    }
}
