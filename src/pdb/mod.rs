//! # `pdb` PDB parsing, modification and writing
//!
//! This module groups all operations that touch the protein structure file:
//!
//! | Sub-module | Responsibility |
//! |---|---|
//! | [`parser`] | Load the reference PDB **once** into an immutable template |
//! | [`deuteration`] | Apply a binary deuteration pattern (non-labile H → D) |
//! | [`d2o`] | Apply a solvent D₂O percentage (labile H → D) |
//!
//! ## Typical call sequence (one iteration of the pipeline)
//!
//! ```text
//! // --- done once at program start ---
//! let template = parser::load_template("protein.pdb")?;
//!
//! // --- repeated for every (pattern, %D₂O) pair ---
//! let deuterated = deuteration::apply_pattern(&template, pattern, mode)?;
//! let ready = d2o::apply_percentage(&deuterated, pct_d2o)?;
//! pdbrust::write_pdb_file(&ready, "pdbs_0010/30.pdb")?;
//! ```
//!
//! ## Hydrogen classification
//!
//! The project distinguishes two classes of hydrogen:
//!
//! * **Non-labile H** — covalently bonded to a **carbon** (`element == "C"` in
//! the preceding atom). These reflect the intrinsic deuteration state of the
//! amino-acid side chain and are changed only by [`deuteration`].
//! * **Labile H** — bonded to **O, N, or S** (hydroxyl, amide, thiol, …).
//! These exchange rapidly with the solvent and are changed only by [`d2o`].
//!
//! The classification is implemented by [`is_labile`], which is shared between
//! both sub-modules.

pub mod d2o;
pub mod deuteration;
pub mod parser;

// Shared helper

use pdbrust::records::Atom;

/// Returns `true` when `hydrogen` is a **labile** hydrogen, i.e. one that
/// exchanges freely with the solvent.
///
/// # Classification rule
///
/// A hydrogen is labile when the **element** of its true parent heavy atom is
/// O, N, or S. The parent is found by scanning backward in `all_atoms` until the
/// first non-H/D atom, so consecutive hydrogens do not break the classification.
///
/// Conversely, a hydrogen bonded to C is **non-labile**.
///
/// # Parameters
/// - `hydrogen` — the H atom to classify.
/// - `all_atoms` — the **sorted** (by `serial`) slice of all atoms in the
/// structure; this slice must include `hydrogen` itself.
///
/// # Returns
/// - `true` if labile (bonded to O/N/S).
/// - `false` if non-labile (bonded to C) or if no predecessor is found
///   (conservative: treat as non-labile so we never accidentally swap it).
pub fn is_labile(hydrogen: &Atom, all_atoms: &[Atom]) -> bool {
    let h_element = hydrogen.element.to_uppercase();
    if h_element != "H" && h_element != "D" {
        return false;
    }

    let pos = match all_atoms.iter().position(|a| a.serial == hydrogen.serial) {
        Some(i) => i,
        None => return false,
    };

    if pos == 0 {
        return false;
    }

    let parent = (0..pos)
        .rev()
        .map(|i| &all_atoms[i])
        .find(|a| {
            let e = a.element.to_uppercase();
            e != "H" && e != "D"
        });

    let parent = match parent {
        Some(p) => p,
        None => return false,
    };

    matches!(parent.element.to_uppercase().as_str(), "O" | "N" | "S")
}

#[cfg(test)]
mod tests {
    use super::*;
    use pdbrust::records::Atom;

    /// Build a minimal atom with only the fields needed by `is_labile`.
    fn make_atom(serial: i32, element: &str) -> Atom {
        Atom::new(
            serial,
            element.to_string(),
            None,
            "ALA".to_string(),
            "A".to_string(),
            1,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            element.to_string(),
            None,
        )
    }

    #[test]
    fn h_after_carbon_is_non_labile() {
        let atoms = vec![make_atom(1, "C"), make_atom(2, "H")];
        assert!(!is_labile(&atoms[1], &atoms));
    }

    #[test]
    fn h_after_nitrogen_is_labile() {
        let atoms = vec![make_atom(1, "N"), make_atom(2, "H")];
        assert!(is_labile(&atoms[1], &atoms));
    }

    #[test]
    fn h_after_oxygen_is_labile() {
        let atoms = vec![make_atom(1, "O"), make_atom(2, "H")];
        assert!(is_labile(&atoms[1], &atoms));
    }

    #[test]
    fn h_after_sulfur_is_labile() {
        let atoms = vec![make_atom(1, "S"), make_atom(2, "H")];
        assert!(is_labile(&atoms[1], &atoms));
    }

    #[test]
    fn h_with_no_predecessor_is_non_labile() {
        let atoms = vec![make_atom(5, "H")];
        assert!(!is_labile(&atoms[0], &atoms));
    }
}