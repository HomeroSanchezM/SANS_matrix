//! # `pdb::d2o`  Solvent D₂O percentage application
//!
//! This module implements **Stage 3** of the per-pattern pipeline:
//! given a structure that has already been through [`deuteration::apply_pattern`],
//! produce `N` copies — one per D₂O percentage step — where a **randomly chosen**
//! fraction of labile hydrogens has been converted to deuterium.
//!
//! ## Physical meaning
//!
//! In a SANS experiment performed in a D₂O/H₂O buffer:
//! - **Non-labile H** (on C) do not exchange with the solvent → already handled
//!   by [`deuteration`](super::deuteration).
//! - **Labile H** (on O, N, S) exchange rapidly with the solvent. At bulk
//!   fraction `X % D₂O`, statistically `X %` of every labile site carries D
//!   at any given instant — **uniformly distributed** across the structure.
//!
//! At `0 % D₂O`   → all labile H remain H.
//! At `100 % D₂O` → all labile H become D.
//! At `X %`        → exactly `floor(n_labile × X / 100)` labile H become D,
//!                   chosen **uniformly at random** across the whole structure.
//!
//! ## Why probabilistic instead of serial-order?
//!
//! The previous deterministic implementation always converted the labile H with
//! the lowest serial numbers first — a physically unrealistic choice that would
//! cluster deuterium labels near the N-terminus. Real solvent exchange is
//! uniform: every labile site has the same probability of carrying D,
//! regardless of its position in the sequence.
//!
//! ## Algorithm — partial Fisher-Yates shuffle
//!
//! [`apply_percentage`] collects the indices of all labile H into a `Vec`,
//! then calls [`rand::seq::SliceRandom::partial_shuffle`] to pick exactly `k`
//! of them at random in **O(k)** time without any extra allocation. The chosen
//! indices are moved to the end of the slice by the shuffle; we convert only
//! those atoms.
//!
//! ## Reproducibility
//!
//! [`apply_percentage`] accepts any [`rand::Rng`] so the caller controls the
//! RNG lifecycle. For reproducible output (tests, benchmarks) pass a seeded
//! [`rand::rngs::StdRng`]. For production runs pass `rand::thread_rng()`.
//! Advancing the same RNG across successive `(pattern, pct)` calls avoids
//! correlation between consecutive structures.
//!
//! ## D₂O step grid
//!
//! The default grid is 0, 5, 10, …, 100 (21 points). The step size is
//! controlled by the `--d2o-step` CLI parameter. [`generate_all_percentages`]
//! builds the full grid from a step value.
//!
//! ## Required Cargo dependency
//!
//! ```toml
//! [dependencies]
//! rand = "0.8"
//! ```

use anyhow::{bail, Result};
use pdbrust::{records::Atom, PdbStructure};
use rand::{seq::SliceRandom, Rng};

use crate::pdb::is_labile;

//  Public API

/// Returns the full list of D₂O percentages for a given step.
///
/// For `step_pct = 5` the output is `[0, 5, 10, …, 100]` (21 values).
/// The value `100` is always included as the last entry even when the step
/// does not divide 100 evenly (e.g. step=30 → `[0, 30, 60, 90, 100]`).
///
/// # Errors
///
/// Returns an error if `step_pct == 0` or `step_pct > 100`.
pub fn generate_all_percentages(step_pct: u8) -> Result<Vec<u8>> {
    if step_pct == 0 || step_pct > 100 {
        bail!("D₂O step must be between 1 and 100 (got {})", step_pct);
    }
    let mut pcts = Vec::new();
    let mut current: u32 = 0;
    while current <= 100 {
        pcts.push(current as u8);
        current += step_pct as u32;
    }
    if pcts.last().copied() != Some(100) {
        pcts.push(100);
    }
    Ok(pcts)
}

/// Applies a single D₂O percentage to a **pre-deuterated** structure,
/// converting labile H atoms to D **uniformly at random**.
///
/// Exactly `floor(n_labile × pct_d2o / 100)` labile H atoms are converted.
/// The specific atoms are chosen by a partial Fisher-Yates shuffle so that
/// every subset of the correct size has equal probability — matching the
/// physical reality of fast, uniform solvent exchange.
///
/// The input `structure` is cloned; the caller's copy is never modified.
///
/// # Parameters
/// - `structure` — structure after [`deuteration::apply_pattern`], sorted by serial.
/// - `pct_d2o`   — solvent D₂O percentage in `[0, 100]`.
/// - `rng`       — random number generator; use a seeded RNG for reproducibility.
///
/// # Errors
///
/// Returns an error if `pct_d2o > 100`.
///
/// # Example
///
/// ```no_run
/// use rand::SeedableRng;
/// use rand::rngs::StdRng;
///
/// let mut rng = StdRng::seed_from_u64(42);
///
/// for pct in generate_all_percentages(5)? {
///     let ready = apply_percentage(&deuterated, pct, &mut rng)?;
///     pdbrust::write_pdb_file(&ready, &format!("pdbs_{pattern}/{pct}.pdb"))?;
/// }
/// ```
pub fn apply_percentage<R: Rng>(
    structure: &PdbStructure,
    pct_d2o: u8,
    rng: &mut R,
) -> Result<PdbStructure> {
    if pct_d2o > 100 {
        bail!("D₂O percentage must be ≤ 100 (got {})", pct_d2o);
    }

    // Fast path: 0 % → nothing to convert.
    if pct_d2o == 0 {
        return Ok(structure.clone());
    }

    let mut modified = structure.clone();

    //  Collect indices of all labile H atoms
    //
    // We pass `structure.atoms` (the original, pre-clone) as the reference
    // slice for `is_labile` because borrowing `modified.atoms` mutably (later)
    // and immutably (here) at the same time is not allowed by the borrow
    // checker. The two slices are identical at this point.
    let mut labile_indices: Vec<usize> = modified
        .atoms
        .iter()
        .enumerate()
        .filter(|(_i, atom)| {
            atom.element.to_uppercase() == "H" && is_labile(atom, &structure.atoms)
        })
        .map(|(i, _atom)| i)
        .collect();

    let n_labile = labile_indices.len();

    // How many labile H to convert?
    // Integer arithmetic: floor(n × pct / 100) — no floating-point rounding.
    let n_to_convert = (n_labile as u64 * pct_d2o as u64 / 100) as usize;

    // Fast path: nothing rounds to zero, or no labile H at all.
    if n_to_convert == 0 {
        return Ok(modified);
    }

    // Fast path: convert every labile H (100 % case, or tiny structures).
    if n_to_convert == n_labile {
        for &idx in &labile_indices {
            convert_h_to_d(&mut modified.atoms[idx]);
        }
        return Ok(modified);
    }

    //  Probabilistic selection via partial Fisher-Yates shuffle
    //
    // `partial_shuffle(rng, k)` picks `k` elements uniformly at random by
    // swapping them to the END of the slice. It returns `(chosen, rest)`.
    // We convert only the atoms pointed to by the `chosen` half.
    //
    // Complexity: O(k) time, O(1) extra space (swaps in-place).
    let (chosen, _rest) = labile_indices.partial_shuffle(rng, n_to_convert);

    for &idx in chosen.iter() {
        convert_h_to_d(&mut modified.atoms[idx]);
    }

    Ok(modified)
}

//  Internal helpers

/// Converts a single labile hydrogen atom in-place to deuterium.
///
/// Both `element` (used by Pepsi-SANS for the bound coherent scattering length)
/// and `name` (for PDB file self-consistency) are updated.
///
/// Name update rule: replace the leading `H` with `D`.
/// - `"H1"` → `"D1"`, `"HE2"` → `"DE2"`, `"H"` → `"D"`.
/// - If the name does not start with `H` (rare edge case), only `element` is
///   updated; the name is left unchanged.
fn convert_h_to_d(atom: &mut Atom) {
    atom.element = "D".to_string();
    if atom.name.starts_with('H') {
        atom.name = format!("D{}", &atom.name[1..]);
    }
}

//  Unit tests

#[cfg(test)]
mod tests {
    use super::*;
    use pdbrust::records::Atom;
    use rand::{rngs::StdRng, SeedableRng};

    fn make_atom(serial: i32, name: &str, element: &str, residue: &str) -> Atom {
        Atom::new(
            serial,
            name.to_string(),
            None,
            residue.to_string(),
            "A".to_string(),
            1,
            0.0, 0.0, 0.0,
            1.0, 0.0,
            element.to_string(),
            None,
        )
    }

    /// Structure with one N (serial 1) and four labile H bonded to it (serials 2-5).
    fn make_four_labile() -> PdbStructure {
        let mut s = PdbStructure::default();
        s.atoms = vec![
            make_atom(1, "N",  "N", "ALA"),
            make_atom(2, "H1", "H", "ALA"),
            make_atom(3, "H2", "H", "ALA"),
            make_atom(4, "H3", "H", "ALA"),
            make_atom(5, "H4", "H", "ALA"),
        ];
        s
    }

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    //  generate_all_percentages

    #[test]
    fn step5_gives_21_points() {
        let pcts = generate_all_percentages(5).unwrap();
        assert_eq!(pcts.len(), 21);
        assert_eq!(pcts[0], 0);
        assert_eq!(pcts[20], 100);
    }

    #[test]
    fn step_30_always_ends_at_100() {
        let pcts = generate_all_percentages(30).unwrap();
        assert_eq!(*pcts.last().unwrap(), 100);
    }

    #[test]
    fn step0_is_error() {
        assert!(generate_all_percentages(0).is_err());
    }

    #[test]
    fn step_over_100_is_error() {
        assert!(generate_all_percentages(101).is_err());
    }

    //  apply_percentage — boundary cases

    #[test]
    fn zero_percent_changes_nothing() {
        let s = make_four_labile();
        let result = apply_percentage(&s, 0, &mut seeded_rng()).unwrap();
        let h_count = result.atoms.iter().filter(|a| a.element == "H").count();
        assert_eq!(h_count, 4);
    }

    #[test]
    fn hundred_percent_converts_all_labile() {
        let s = make_four_labile();
        let result = apply_percentage(&s, 100, &mut seeded_rng()).unwrap();
        let d_count = result.atoms.iter().filter(|a| a.element == "D").count();
        assert_eq!(d_count, 4, "all 4 labile H must become D at 100 %");
    }

    #[test]
    fn pct_over_100_is_error() {
        let s = make_four_labile();
        assert!(apply_percentage(&s, 101, &mut seeded_rng()).is_err());
    }

    #[test]
    fn source_structure_is_not_mutated() {
        let s = make_four_labile();
        let _result = apply_percentage(&s, 100, &mut seeded_rng()).unwrap();
        let h_count = s.atoms.iter().filter(|a| a.element == "H").count();
        assert_eq!(h_count, 4, "original must be unchanged");
    }

    //  apply_percentage — count correctness

    #[test]
    fn correct_count_converted_at_50_pct() {
        // 4 labile H, 50 % → floor(4 × 50 / 100) = 2 converted.
        let s = make_four_labile();
        let result = apply_percentage(&s, 50, &mut seeded_rng()).unwrap();
        let d_count = result.atoms.iter().filter(|a| a.element == "D").count();
        assert_eq!(d_count, 2);
    }

    #[test]
    fn correct_count_converted_at_75_pct() {
        // 4 labile H, 75 % → floor(4 × 75 / 100) = 3 converted.
        let s = make_four_labile();
        let result = apply_percentage(&s, 75, &mut seeded_rng()).unwrap();
        let d_count = result.atoms.iter().filter(|a| a.element == "D").count();
        assert_eq!(d_count, 3);
    }

    //  apply_percentage — probabilistic properties

    #[test]
    fn no_atom_is_converted_twice() {
        // All converted atoms must have distinct serials.
        let s = make_four_labile();
        let result = apply_percentage(&s, 75, &mut seeded_rng()).unwrap();
        let d_serials: Vec<i32> = result
            .atoms
            .iter()
            .filter(|a| a.element == "D")
            .map(|a| a.serial)
            .collect();
        let unique: std::collections::HashSet<_> = d_serials.iter().collect();
        assert_eq!(d_serials.len(), unique.len(), "no atom converted twice");
    }

    #[test]
    fn different_seeds_produce_different_selections() {
        // With 4 labile H and k=2, there are C(4,2)=6 possible pairs.
        // Over 20 distinct seeds the selected pair should not always be the same.
        let s = make_four_labile();
        let first_result = {
            let mut rng = StdRng::seed_from_u64(0);
            let r = apply_percentage(&s, 50, &mut rng).unwrap();
            let mut v: Vec<i32> = r
                .atoms
                .iter()
                .filter(|a| a.element == "D")
                .map(|a| a.serial)
                .collect();
            v.sort_unstable();
            v
        };
        let all_same = (1u64..20).all(|seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let r = apply_percentage(&s, 50, &mut rng).unwrap();
            let mut v: Vec<i32> = r
                .atoms
                .iter()
                .filter(|a| a.element == "D")
                .map(|a| a.serial)
                .collect();
            v.sort_unstable();
            v == first_result
        });
        assert!(!all_same, "random selection must vary across seeds");
    }

    #[test]
    fn all_labile_sites_are_reachable() {
        // Over 40 seeds every labile H (serials 2-5) should be chosen at least once.
        // At k=2 out of 4, each site has 50 % chance per draw; 40 draws makes
        // the probability of missing any site negligible.
        let s = make_four_labile();
        let mut seen: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for seed in 0u64..40 {
            let mut rng = StdRng::seed_from_u64(seed);
            let r = apply_percentage(&s, 50, &mut rng).unwrap();
            for atom in r.atoms.iter().filter(|a| a.element == "D") {
                seen.insert(atom.serial);
            }
        }
        for serial in [2, 3, 4, 5] {
            assert!(
                seen.contains(&serial),
                "serial {serial} was never selected — distribution appears biased"
            );
        }
    }

    //  convert_h_to_d

    #[test]
    fn h1_becomes_d1() {
        let mut atom = make_atom(1, "H1", "H", "ALA");
        convert_h_to_d(&mut atom);
        assert_eq!(atom.element, "D");
        assert_eq!(atom.name, "D1");
    }

    #[test]
    fn bare_h_becomes_d() {
        let mut atom = make_atom(1, "H", "H", "ALA");
        convert_h_to_d(&mut atom);
        assert_eq!(atom.name, "D");
    }
}
