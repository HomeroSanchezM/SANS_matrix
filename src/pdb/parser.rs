//! # `pdb::parser`  Reference PDB loader
//!
//! This module is responsible for **one task only**: read the reference `.pdb`
//! file from disk a single time and return an owned [`PdbStructure`] that
//! serves as the immutable template for the rest of the pipeline.
//!
//! ## Why parse only once?
//!
//! The pipeline generates up to 262 144 deuteration patterns.  Each pattern
//! starts from the same original atomic coordinates.  Parsing the file on
//! every iteration would be wasteful; instead:
//!
//! 1. [`load_template`] is called **once** at program startup.
//! 2. The returned `PdbStructure` is kept in memory for the duration of the
//!    run.
//! 3. Every call to [`deuteration::apply_pattern`] or [`d2o::apply_percentage`]
//!    clones only the atoms it needs to modify, leaving the template untouched.
//!
//! ## Atom ordering guarantee
//!
//! Both [`deuteration`](super::deuteration) and [`d2o`](super::d2o) rely on
//! atoms being sorted by `serial` number so that predecessor look-ups in
//! [`is_labile`](super::is_labile) are correct.  [`load_template`] enforces
//! this order before returning.
//!
//! ## Only ATOM records
//!
//! HETATM records (waters, ligands, ions) are excluded because Pepsi-SANS
//! models only the protein chain and because deuterating water molecules
//! is handled separately via the `--d2o` flag passed to Pepsi-SANS itself.

use anyhow::{Context, Result};
use pdbrust::{parse_pdb_file, PdbStructure};
use std::path::Path;

//  Public API

/// Loads the reference PDB file and returns an immutable template structure.
///
/// The function:
/// 1. Calls [`pdbrust::parse_pdb_file`] to deserialise all records.
/// 2. Discards HETATM atoms (waters, ligands, ions).
/// 3. Sorts the remaining ATOM records by `serial` number (ascending).
/// 4. Validates that at least one atom was loaded.
///
/// # Errors
///
/// Returns an error if:
/// - The file does not exist or cannot be opened.
/// - The PDB content is malformed and pdbrust cannot parse it.
/// - The file contains **no** protein ATOM records after filtering.
///
/// # Example
///
/// ```no_run
/// let template = parser::load_template("data/protein.pdb")?;
/// println!("Loaded {} protein atoms", template.atoms.len());
/// ```
pub fn load_template(path: impl AsRef<Path>) -> Result<PdbStructure> {
    let path = path.as_ref();

    //  Step 1: Parse the file with pdbrust
    // `parse_pdb_file` returns Result<PdbStructure, PdbError>.
    // We convert the PdbError to anyhow and attach a human-readable context.
    let mut structure = parse_pdb_file(path)
        .with_context(|| format!("Failed to parse PDB file: {}", path.display()))?;

    //  Step 2: Remove HETATM records
    // `atom.is_hetatm` is true for ligands, waters (HOH), and ions.
    // Pepsi-SANS operates on the protein chain only; keeping HETATM records
    // would introduce noise in the scattering calculation.
    let before = structure.atoms.len();
    structure.atoms.retain(|atom| !atom.is_hetatm);
    let removed = before - structure.atoms.len();
    if removed > 0 {
        eprintln!(
            "[parser] Removed {removed} HETATM atoms from '{}'",
            path.display()
        );
    }

    //  Step 3: Sort atoms by serial number
    // The `is_labile` helper (pdb/mod.rs) walks the atom list sequentially to
    // find the predecessor of each H.  Sorting here guarantees the expected
    // predecessor relationship: heavy atom immediately before its H.
    structure.atoms.sort_unstable_by_key(|a| a.serial);

    // Step 4: Validate that we have something to work with
    anyhow::ensure!(
        !structure.atoms.is_empty(),
        "PDB file '{}' contains no protein ATOM records",
        path.display()
    );

    eprintln!(
        "[parser] Loaded {} protein atoms from '{}'",
        structure.atoms.len(),
        path.display()
    );

    Ok(structure)
}

//  Unit tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Writes a minimal valid PDB string to a temp file and returns the path.
    fn write_temp_pdb(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }

    /// A minimal two-atom PDB: one ATOM (CA of ALA) and one HETATM (HOH).
    const MINIMAL_PDB: &str = "\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
HETATM    2  O   HOH A 100       5.000   5.000   5.000  1.00  0.00           O
END
";

    #[test]
    fn loads_atom_and_strips_hetatm() {
        let f = write_temp_pdb(MINIMAL_PDB);
        let tmpl = load_template(f.path()).unwrap();
        // Only the ATOM record should survive.
        assert_eq!(tmpl.atoms.len(), 1);
        assert!(!tmpl.atoms[0].is_hetatm);
        assert_eq!(tmpl.atoms[0].residue_name, "ALA");
    }

    #[test]
    fn atoms_are_sorted_by_serial() {
        // Give the atoms reversed serial numbers in the file.
        let pdb = "\
ATOM      2  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
END
";
        let f = write_temp_pdb(pdb);
        let tmpl = load_template(f.path()).unwrap();
        assert_eq!(tmpl.atoms[0].serial, 1);
        assert_eq!(tmpl.atoms[1].serial, 2);
    }

    #[test]
    fn empty_protein_returns_error() {
        // File contains only a HETATM — no ATOM records survive.
        let pdb = "HETATM    1  O   HOH A 100       5.000   5.000   5.000  1.00  0.00           O\nEND\n";
        let f = write_temp_pdb(pdb);
        assert!(load_template(f.path()).is_err());
    }

    #[test]
    fn nonexistent_file_returns_error() {
        assert!(load_template("/nonexistent/path/protein.pdb").is_err());
    }
}
