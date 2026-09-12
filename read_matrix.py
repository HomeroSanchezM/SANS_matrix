#!/usr/bin/env python3
"""
Lecture, vérification et calcul de fitness pour les matrices HDF5 OptiSANS.

Compatible avec:
  - matrices brutes de main_matrix.py
  - matrices après ajustement B-spline (bspline_fit_rg_Imax_filter.py)

Usage:
    python read_matrix.py matrix.h5 --list
    python read_matrix.py matrix.h5 --print
    python read_matrix.py matrix.h5 --gamma 2
    python read_matrix.py matrix.h5 --gamma 2 --csv fitness_out.csv
    python read_matrix.py matrix.h5 --aa PRO MET LYS
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np


REQUIRED_DATASETS = {
    "0": "patterns",
    "1": "d2o_pct",
    "2": "ratio",
    "3": "product_areas",
    "4": "Q",
    "5": "I",
    "6": "deuterium_pct",
    "9": "nonlabile_deut_pct",
    "10": "rg",
}

PARAM_PRINT_ORDER = [
    "patterns",
    "d2o_pct",
    "ratio",
    "product_areas",
    "deuterium_pct",
    "nonlabile_deut_pct",
    "rg",
]

# Largeurs de colonnes pour l'affichage.
COLUMN_WIDTH = {
    "patterns": 12,
    "d2o_pct": 8,
    "ratio": 14,
    "product_areas": 14,
    "deuterium_pct": 16,
    "nonlabile_deut_pct": 16,
    "rg": 10,
}

# ----------------------------------------------------------------------
# Groupes d'acides aminés — DOIVENT rester identiques à main_matrix.py.
# ----------------------------------------------------------------------
COMPLETE_GROUPS = [
    {"label": "ALA", "residues": ["ALA"]},
    {"label": "ARG", "residues": ["ARG"]},
    {"label": "ASN/ASP", "residues": ["ASN", "ASP"]},
    {"label": "CYS", "residues": ["CYS"]},
    {"label": "GLU/GLN", "residues": ["GLU", "GLN"]},
    {"label": "GLY", "residues": ["GLY"]},
    {"label": "HIS", "residues": ["HIS"]},
    {"label": "ILE", "residues": ["ILE"]},
    {"label": "LEU", "residues": ["LEU"]},
    {"label": "LYS", "residues": ["LYS"]},
    {"label": "MET", "residues": ["MET"]},
    {"label": "PHE", "residues": ["PHE"]},
    {"label": "PRO", "residues": ["PRO"]},
    {"label": "SER", "residues": ["SER"]},
    {"label": "THR", "residues": ["THR"]},
    {"label": "TRP", "residues": ["TRP"]},
    {"label": "TYR", "residues": ["TYR"]},
    {"label": "VAL", "residues": ["VAL"]},
]

REDUCED_GROUPS = [
    {"label": "ARG", "residues": ["ARG"]},
    {"label": "GLU/GLN", "residues": ["GLU", "GLN"]},
    {"label": "HIS", "residues": ["HIS"]},
    {"label": "ILE", "residues": ["ILE"]},
    {"label": "LEU", "residues": ["LEU"]},
    {"label": "LYS", "residues": ["LYS"]},
    {"label": "PHE", "residues": ["PHE"]},
    {"label": "PRO", "residues": ["PRO"]},
    {"label": "THR", "residues": ["THR"]},
    {"label": "TYR", "residues": ["TYR"]},
    {"label": "VAL", "residues": ["VAL"]},
]


# ----------------------------------------------------------------------
# Utilitaires de patterns AA
# ----------------------------------------------------------------------
def pattern_string(bits: int, n_bits: int) -> str:
    return format(bits, f'0{n_bits}b')


def infer_groups(n_bits: int):
    if n_bits == len(REDUCED_GROUPS):
        return REDUCED_GROUPS, "reduced"
    if n_bits == len(COMPLETE_GROUPS):
        return COMPLETE_GROUPS, "complete"
    raise ValueError(
        f"Impossible de déterminer le mode pour des patterns de {n_bits} bits : "
        f"attendu {len(REDUCED_GROUPS)} (reduced) ou {len(COMPLETE_GROUPS)} (complete)."
    )


def find_group_index(aa: str, groups: list) -> int | None:
    aa_upper = aa.strip().upper()
    for i, g in enumerate(groups):
        if aa_upper in [r.upper() for r in g["residues"]]:
            return i
    return None


# ----------------------------------------------------------------------
# Gestion des colonnes fitness_gamma_*
# ----------------------------------------------------------------------
def dataset_name_for_gamma(gamma: float) -> str:
    s = f"{gamma:g}"
    s = s.replace('-', 'neg').replace('.', 'p')
    return f"fitness_gamma_{s}"


def list_fitness_columns(f: h5py.File) -> list:
    names = [k for k in f.keys() if k.startswith("fitness_gamma_")]

    def gamma_of(name: str) -> float:
        s = name[len("fitness_gamma_"):]
        s = s.replace('neg', '-').replace('p', '.')
        try:
            return float(s)
        except ValueError:
            return float('inf')

    return sorted(names, key=gamma_of)


def compute_fitness(product_areas, ratio, gamma):
    return np.asarray(product_areas, dtype=float) * (
        np.asarray(ratio, dtype=float) ** float(gamma)
    )


def store_fitness_column(path: Path, gamma: float) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Fichier HDF5 introuvable: {path}")

    ds_name = dataset_name_for_gamma(gamma)
    with h5py.File(path, "r+") as f:
        if "product_areas" not in f or "ratio" not in f:
            raise ValueError(
                "Le fichier ne contient pas 'product_areas'/'ratio' : "
                "impossible de calculer la fitness."
            )
        product_areas = np.array(f["product_areas"])
        ratios = np.array(f["ratio"])
        fitness_values = compute_fitness(product_areas, ratios, gamma).astype(np.float32)

        if ds_name in f:
            f[ds_name][...] = fitness_values
            print(f"  Colonne '{ds_name}' mise à jour (gamma={gamma:g}).")
        else:
            f.create_dataset(ds_name, data=fitness_values, dtype='<f4')
            print(f"  Colonne '{ds_name}' créée (gamma={gamma:g}).")

    return ds_name


# ----------------------------------------------------------------------
# Structure / --list
# ----------------------------------------------------------------------
def check_structure(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Fichier HDF5 introuvable: {path}")

    info = {}
    with h5py.File(path, "r") as f:
        # datasets standards
        for key, name in REQUIRED_DATASETS.items():
            if name in f:
                ds = f[name]
                info[key] = (True, ds.shape, ds.dtype)
            else:
                info[key] = (False, None, None)

        if "Q" not in f:
            raise ValueError("Dataset 'Q' absent: la matrice semble invalide.")

        q_ds = f["Q"]
        info["Q"] = (True, q_ds.shape, q_ds.dtype)
        info["n_q"] = q_ds.shape[0]

        # nombre de courbes : préférer bspline_mask si présent, sinon I
        if "bspline_mask" in f:
            info["n_curves"] = int(f["bspline_mask"].shape[0])
        elif "I" in f:
            info["n_curves"] = int(f["I"].shape[0])
        else:
            info["n_curves"] = None

        if "I" in f:
            info["n_points"] = int(f["I"].shape[1])
        else:
            info["n_points"] = None

        if "patterns" in f:
            info["n_patterns"] = int(np.asarray(f["patterns"]).shape[0])
        else:
            info["n_patterns"] = None

        # champs B-spline
        for bs_name in ("bspline_coefs", "bspline_mask", "bspline_nbasis"):
            if bs_name in f:
                ds = f[bs_name]
                info[bs_name] = (True, ds.shape, ds.dtype)
            else:
                info[bs_name] = (False, None, None)

        # colonnes fitness_gamma_*
        info["fitness_columns"] = {}
        for name in list_fitness_columns(f):
            ds = f[name]
            info["fitness_columns"][name] = (ds.shape, ds.dtype)

    return info


def _present_shape_dtype(info: dict, name: str) -> str:
    if name not in info:
        return "absent"
    present, shape, dtype = info[name]
    if not present:
        return "absent"
    return f"shape={shape}, dtype={dtype}"


def print_structure(info: dict) -> None:
    print("== Structure de la matrice ==")
    for key, name in REQUIRED_DATASETS.items():
        present, shape, dtype = info[key]
        status = "OK" if present else "ABSENT"
        detail = _present_shape_dtype(info, name)
        print(f"  ['{key}'] {name:<20s} {status:<8s} {detail}")

    print(f"\n  n_q       : {info.get('n_q')}")
    print(f"  n_curves  : {info.get('n_curves')}")
    print(f"  n_points  : {info.get('n_points')}")
    print(f"  n_patterns: {info.get('n_patterns')}")

    bs_keys = ("bspline_coefs", "bspline_mask", "bspline_nbasis")
    if any(info.get(k, (False, None, None))[0] for k in bs_keys):
        print("\n  == Champs B-spline ==")
        for name in bs_keys:
            print(f"  {name:<20s} {_present_shape_dtype(info, name)}")

    fitness_cols = info.get("fitness_columns", {})
    if fitness_cols:
        print("\n  == Colonnes de fitness (gamma) ==")
        for name, (shape, dtype) in fitness_cols.items():
            print(f"  {name:<25s} OK       shape={shape}, dtype={dtype}")
    else:
        print("\n  (aucune colonne de fitness enregistrée -- utilisez --gamma X pour en créer une)")


# ----------------------------------------------------------------------
# --print : affichage des 10 premières lignes
# ----------------------------------------------------------------------
def _column_width(name: str) -> int:
    return COLUMN_WIDTH.get(name, max(16, len(name) + 2))


def _format_cell(name: str, value) -> str:
    width = _column_width(name)
    if name == "patterns":
        s = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        return f"{s:>{width}s}"
    if name == "d2o_pct":
        return f"{int(value):>{width}d}"
    if name == "rg":
        return f"{float(value):>{width}.4f}"
    return f"{float(value):>{width}.6e}"


def _print_table(f: h5py.File, indices, columns: list, title: str) -> None:
    print(f"== {title} ==\n")
    header = f"{'idx':>4s}  " + "  ".join(f"{c:>{_column_width(c)}s}" for c in columns)
    print(header)
    print("-" * len(header))
    for i in indices:
        cells = [_format_cell(c, f[c][i]) for c in columns]
        print(f"{i:4d}  " + "  ".join(cells))


def _display_columns(f: h5py.File) -> list:
    """Colonnes de base + fitness_gamma_* si présentes."""
    columns = [c for c in PARAM_PRINT_ORDER if c in f]
    columns.extend(list_fitness_columns(f))
    return columns


def cmd_list(args: argparse.Namespace) -> int:
    path = Path(args.matrix)
    info = check_structure(path)
    print_structure(info)
    return 0


def print_first_lines(path: Path, max_rows: int = 10) -> int:
    with h5py.File(path, "r") as f:
        if "bspline_mask" in f:
            n_curves = int(f["bspline_mask"].shape[0])
        elif "I" in f:
            n_curves = int(f["I"].shape[0])
        elif "patterns" in f:
            n_curves = int(np.asarray(f["patterns"]).shape[0])
        else:
            raise ValueError("Impossible de déterminer le nombre de courbes.")

        n = min(max_rows, n_curves)
        columns = _display_columns(f)
        _print_table(f, range(n), columns, f"{n} premières lignes des paramètres (sur {n_curves})")
    return 0


# ----------------------------------------------------------------------
# --aa : filtrer par pattern correspondant à une liste d'AA
# ----------------------------------------------------------------------
def resolve_aa_pattern(f: h5py.File, aa_list: list):
    if "patterns" not in f:
        raise ValueError("Dataset 'patterns' absent du fichier.")

    first_pattern = f["patterns"][0]
    first_pattern = (
        first_pattern.decode("utf-8") if isinstance(first_pattern, bytes) else str(first_pattern)
    )
    n_bits = len(first_pattern)
    groups, mode_name = infer_groups(n_bits)

    bits = 0
    unresolved = []
    for aa in aa_list:
        idx = find_group_index(aa, groups)
        if idx is None:
            unresolved.append(aa)
        else:
            bits |= (1 << idx)

    if unresolved:
        valid = sorted({res for g in groups for res in g["residues"]})
        raise ValueError(
            f"AA non reconnu(s) pour le mode '{mode_name}' ({n_bits} groupes) : "
            f"{', '.join(unresolved)}. AA valides: {', '.join(valid)}"
        )

    return pattern_string(bits, n_bits), mode_name, groups


def cmd_aa(args: argparse.Namespace) -> int:
    path = Path(args.matrix)
    aa_list = [a.upper() for a in args.aa]

    with h5py.File(path, "r") as f:
        try:
            target_pattern, mode_name, groups = resolve_aa_pattern(f, aa_list)
        except ValueError as e:
            print(f"Erreur: {e}", file=sys.stderr)
            return 1

        patterns_raw = f["patterns"][:]
        patterns_decoded = [
            p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in patterns_raw
        ]
        indices = [i for i, p in enumerate(patterns_decoded) if p == target_pattern]

        print(f"AA demandés   : {', '.join(aa_list)}")
        print(f"Mode détecté  : {mode_name} ({len(groups)} groupes)")
        print(f"Pattern cible : {target_pattern}\n")

        if not indices:
            print("Aucune courbe ne correspond à ce pattern "
                  "(tous les autres AA sont considérés protonés).")
            return 0

        columns = _display_columns(f)
        _print_table(f, indices, columns, f"{len(indices)} courbe(s) trouvée(s)")

    return 0


# ----------------------------------------------------------------------
# --gamma : résumé + CSV, et enregistrement dans le HDF5
# ----------------------------------------------------------------------
def cmd_fitness(args: argparse.Namespace, gamma: float) -> int:
    path = Path(args.matrix)

    print(f"== Calcul de la fitness (gamma={gamma:g}) ==")
    print(f"  Matrice: {path}")

    if not path.exists():
        print("Erreur: fichier introuvable.", file=sys.stderr)
        return 1

    with h5py.File(path, "r") as f:
        if "product_areas" not in f or "ratio" not in f:
            print("Erreur: 'product_areas'/'ratio' absents du fichier.", file=sys.stderr)
            return 1

        n_curves = int(f["product_areas"].shape[0])
        print(f"  Courbes : {n_curves}")

        ratios = np.array(f["ratio"])
        product_areas = np.array(f["product_areas"])

    fitness_values = compute_fitness(product_areas, ratios, gamma)

    print(f"\n  Fitness min : {np.min(fitness_values):.6e}")
    print(f"  Fitness max : {np.max(fitness_values):.6e}")
    print(f"  Fitness mean: {np.mean(fitness_values):.6e}")
    n_curves = int(ratios.shape[0])
    print(f"  Courbes avec ratio < 0.01 : {int(np.sum(ratios < 0.01))} / {n_curves}")
    print(
        f"  Courbes avec fitness == 0 (ratio check failed): "
        f"{int(np.sum(fitness_values == 0.0))} / {n_curves}"
    )

    if args.csv:
        csv_path = Path(args.csv)
        with h5py.File(path, "r") as f:
            patterns = np.array(f["patterns"])
            d2o_pct = np.array(f["d2o_pct"])
            deuterium_pct = np.array(f["deuterium_pct"])
            nonlabile_deut_pct = np.array(f["nonlabile_deut_pct"])
            rg = np.array(f["rg"])

            # Colonnes B-spline et fitness_gamma_* si présentes
            extra_columns = []
            for bs_name in ("bspline_coefs", "bspline_mask", "bspline_nbasis"):
                if bs_name in f:
                    extra_columns.append(bs_name)
            extra_columns.extend(list_fitness_columns(f))

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            header = [
                "index", "pattern", "d2o_pct", "deuterium_pct", "nonlabile_deut_pct",
                "rg", "ratio", "product_areas", "fitness",
            ]
            header.extend(extra_columns)
            writer.writerow(header)

            for i in range(len(ratios)):
                row = [
                    i,
                    patterns[i].decode("utf-8") if isinstance(patterns[i], bytes) else patterns[i],
                    int(d2o_pct[i]),
                    float(deuterium_pct[i]),
                    float(nonlabile_deut_pct[i]),
                    float(rg[i]),
                    float(ratios[i]),
                    float(product_areas[i]),
                    float(fitness_values[i]),
                ]
                with h5py.File(path, "r") as f:
                    for col in extra_columns:
                        row.append(float(f[col][i]))
                writer.writerow(row)
        print(f"\n  CSV enregistré: {csv_path}")

    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lecture et vérification de la matrice binaire HDF5 OptiSANS."
    )
    parser.add_argument("matrix", type=str, help="Chemin vers le fichier HDF5")
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help=(
            "Exposant gamma pour fitness = product_areas * ratio^gamma. "
            "Calcule la fitness ET l'enregistre dans le HDF5 sous "
            "'fitness_gamma_<gamma>' (defaut si absent: 2, sans écriture)."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Afficher la structure de la matrice",
    )
    parser.add_argument(
        "--print",
        dest="three",
        action="store_true",
        help="Afficher les 10 premières lignes des paramètres",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Enregistrer les résultats dans un CSV",
    )
    parser.add_argument(
        "--aa",
        nargs="+",
        metavar="AA",
        default=None,
        help=(
            "Liste de codes AA 3 lettres (ex: --aa PRO MET LYS) : reconstruit "
            "le pattern binaire correspondant (ces AA deutérés, tous les "
            "autres protonés) et affiche les courbes associées."
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = Path(args.matrix)

    # Si --gamma est explicitement fourni, on calcule et enregistre la colonne
    # dans le HDF5 quelle que soit l'action demandée par ailleurs.
    gamma_explicit = args.gamma is not None
    if gamma_explicit:
        try:
            store_fitness_column(path, args.gamma)
        except (FileNotFoundError, ValueError) as e:
            print(f"Erreur: {e}", file=sys.stderr)
            return 1

    if args.aa:
        return cmd_aa(args)

    if args.three:
        return print_first_lines(path, max_rows=10)

    if args.list:
        return cmd_list(args)

    # Action par défaut : résumé de fitness
    effective_gamma = args.gamma if gamma_explicit else 2.0
    return cmd_fitness(args, effective_gamma)


if __name__ == "__main__":
    sys.exit(main())
