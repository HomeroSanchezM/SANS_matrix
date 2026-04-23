//! # `pdb::deuteration` Non-labile hydrogen substitution
//!
//! This module implements **Stage 2** of the per-pattern pipeline:
//! given the immutable PDB template and a [`DeuterationPattern`], produce a
//! new [`PdbStructure`] where every non-labile hydrogen (`H`) belonging to a
//! **deuterated group** has been renamed to deuterium (`D`).
//!
//! ## What changes, what stays the same
//!
//! | Atom class | Action |
//! |---|---|
//! | Heavy atoms (C, N, O, S, …) | Cloned verbatim — coordinates unchanged |
//! | Non-labile H in a **deuterated** group | `element` and `name` set to `"D"` |
//! | Non-labile H in a **protonated** group | Cloned verbatim |
//! | Labile H (bonded to O/N/S) | Cloned verbatim — handled later by [`d2o`] |
//!
//! ## Non-labile hydrogen definition
//!
//! An H is non-labile when its immediate predecessor in the sorted atom list
//! has element `"C"`.  See [`super::is_labile`] for the full classification
//! logic.
//!
//! ## Group membership
//!
//! An atom belongs to a group if its `residue_name` appears in
//! `group.residues`.  The pattern bit for that group determines whether the
//! group is deuterated.  Groups not present in the template protein are
//! silently skipped (no error).

use anyhow::Result;
use pdbrust::{records::Atom, PdbStructure};

use crate::{
    amino_acids::AaMode,
    binary_vectors::DeuterationPattern,
    pdb::is_labile,
};

//  Public API

/// Applies `pattern` to `template`, returning a **new** structure where
/// non-labile H atoms in deuterated groups have been converted to D.
///
/// The function clones the entire atom list, modifying only the atoms that
/// match the deuteration criteria.  The original `template` is not modified.
///
/// # Parameters
/// - `template`  — the immutable reference structure (sorted by serial).
/// - `pattern`   — which groups are deuterated (one bit per group).
/// - `mode`      — the grouping scheme; determines how the pattern bits map
///                 to amino-acid residue names.
///
/// # Errors
///
/// Currently infallible (`Ok` is always returned), but the signature uses
/// `Result` so that validation logic can be added without a breaking change.
///
/// # Example
///
/// ```no_run
/// let modified = apply_pattern(&template, pattern, AaMode::Reduced)?;
/// pdbrust::write_pdb_file(&modified, "pdbs_00100000000/0.pdb")?;
/// ```
pub fn apply_pattern(
    template: &PdbStructure,
    pattern: DeuterationPattern,
    mode: AaMode,
) -> Result<PdbStructure> {
    // Build a lookup: residue_name → is_deuterated (bool).
    // This avoids scanning the group list for every atom.
    let deuterated_residues = build_deuterated_set(pattern, mode);

    // Clone the structure; we only mutate the atom list.
    let mut modified = template.clone();

    // We need the full sorted atom list as a read-only reference to classify
    // each H with `is_labile`.  Borrow the original template atoms for this —
    // the clone's atom list is identical at this point.
    let reference_atoms: &[Atom] = &template.atoms;

    for atom in modified.atoms.iter_mut() {
        // Only hydrogen atoms can be substituted.
        if atom.element.to_uppercase() != "H" {
            continue;
        }

        // Only atoms in a deuterated residue group.
        if !deuterated_residues.contains(atom.residue_name.as_str()) {
            continue;
        }

        // Only non-labile hydrogens (labile ones are handled by `d2o`).
        if is_labile(atom, reference_atoms) {
            continue;
        }

        // All checks passed: convert H → D.
        substitute_h_to_d(atom);
    }

    Ok(modified)
}

//  Internal helpers

/// Builds the set of residue names that are **deuterated** under `pattern`.
///
/// Returns a `HashSet<&'static str>` because residue names in the group
/// definitions are `'static` string literals.
fn build_deuterated_set(
    pattern: DeuterationPattern,
    mode: AaMode,
) -> std::collections::HashSet<&'static str> {
    mode.groups()
        .iter()
        .enumerate()
        .filter(|(i, _group)| pattern.is_deuterated(*i))
        .flat_map(|(_i, group)| group.residues.iter().copied())
        .collect()
}

/// Converts a single hydrogen atom in-place to deuterium.
///
/// Both the `element` field (used by Pepsi-SANS for the scattering length) and
/// the `name` field (the PDB atom name, e.g. `"HA"` → `"DA"`) are updated so
/// that written PDB files are self-consistent.
///
/// # Atom name update rule
///
/// PDB convention: atom names for hydrogens start with `H`.  We replace the
/// first `H` character with `D` while keeping the rest of the name intact.
/// If the name does not start with `H` (unusual but possible), the name is
/// left unchanged and only `element` is updated.
fn substitute_h_to_d(atom: &mut Atom) {
    // Update the element symbol — this is the field Pepsi-SANS actually reads.
    atom.element = "D".to_string();

    // Update the atom name for PDB file consistency.
    // Example: "HA"  → "DA", "HB2" → "DB2", "H"   → "D"
    if atom.name.starts_with('H') {
        atom.name = format!("D{}", &atom.name[1..]);
    }
    // If name does not start with H (edge case), element update is sufficient.
}

//  Unit tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amino_acids::AaMode;
    use crate::binary_vectors::DeuterationPattern;
    use pdbrust::records::Atom;

    /// Convenience: build an Atom with only the fields we care about.
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

    /// Minimal PdbStructure with two atoms: a CB carbon and one HA hydrogen
    /// (non-labile, bonded to C).
    fn make_template(residue: &str) -> PdbStructure {
        let mut s = PdbStructure::default();
        s.atoms = vec![
            make_atom(1, "CB", "C", residue), // parent carbon
            make_atom(2, "HA", "H", residue), // non-labile H bonded to C
        ];
        s
    }

    //  substitute_h_to_d

    #[test]
    fn h_name_renamed_to_d() {
        let mut atom = make_atom(1, "HA2", "H", "ALA");
        substitute_h_to_d(&mut atom);
        assert_eq!(atom.element, "D");
        assert_eq!(atom.name, "DA2");
    }

    #[test]
    fn single_h_name_becomes_d() {
        let mut atom = make_atom(1, "H", "H", "ALA");
        substitute_h_to_d(&mut atom);
        assert_eq!(atom.name, "D");
    }

    //  apply_pattern  deuterated group

    #[test]
    fn non_labile_h_in_deuterated_group_is_substituted() {
        // Reduced mode, bit 0 = ARG.  Pattern = 0b00000000001 → ARG deuterated.
        let pattern = DeuterationPattern::new_for_test(1, 11);
        let template = make_template("ARG");
        let result = apply_pattern(&template, pattern, AaMode::Reduced).unwrap();

        let h_atoms: Vec<_> = result.atoms.iter().filter(|a| a.element == "H").collect();
        let d_atoms: Vec<_> = result.atoms.iter().filter(|a| a.element == "D").collect();

        assert_eq!(h_atoms.len(), 0, "all H in deuterated ARG should be D");
        assert_eq!(d_atoms.len(), 1);
        assert_eq!(d_atoms[0].name, "DA");
    }

    //  apply_pattern — protonated group

    #[test]
    fn h_in_protonated_group_is_unchanged() {
        // Pattern all zeros: nothing deuterated.
        let pattern = DeuterationPattern::new_for_test(0, 11);
        let template = make_template("ARG");
        let result = apply_pattern(&template, pattern, AaMode::Reduced).unwrap();

        let h_atoms: Vec<_> = result.atoms.iter().filter(|a| a.element == "H").collect();
        assert_eq!(h_atoms.len(), 1, "H should be untouched in protonated group");
    }

    //  apply_pattern — labile H untouched

    #[test]
    fn labile_h_in_deuterated_group_is_not_substituted() {
        // ARG deuterated, but the H is bonded to N (labile).
        let pattern = DeuterationPattern::new_for_test(1, 11); // bit 0 = ARG
        let mut template = make_template("ARG");
        // Replace the parent atom with N so the H becomes labile.
        template.atoms[0].element = "N".to_string();

        let result = apply_pattern(&template, pattern, AaMode::Reduced).unwrap();
        // The labile H must NOT have been converted.
        let h_atoms: Vec<_> = result.atoms.iter().filter(|a| a.element == "H").collect();
        assert_eq!(h_atoms.len(), 1, "labile H must not be changed by deuteration");
    }

    //  template is not modified 

    #[test]
    fn template_is_not_mutated() {
        let pattern = DeuterationPattern::new_for_test(1, 11);
        let template = make_template("ARG");
        let _result = apply_pattern(&template, pattern, AaMode::Reduced).unwrap();
        // Original template must still have H, not D.
        assert_eq!(template.atoms[1].element, "H");
    }
}
