#!/usr/bin/env python3
"""
Cluster average curves plotter.

Lit une matrice HDF5 issue de main_matrix.py / bspline_fit.py dans laquelle
les labels de clustering ont deja ete calcules par spline_clustering.py, puis
genere un figure avec:
  - les courbes individuelles de chaque cluster (translucides)
  - la moyenne du cluster en surimpression (gras)

Memoire: charge un cluster a la fois depuis bspline_coefs ou I.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from skfda.representation.basis import BSplineBasis
from skfda.representation import FDataBasis
from typing import Optional


REQUIRED = ["Q", "patterns"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-cluster curves and cluster averages from HDF5."
    )
    parser.add_argument("h5_path", type=str, help="Chemin vers le fichier HDF5")
    parser.add_argument(
        "--umap",
        action="store_true",
        help="Utiliser umap_hdbscan_labels",
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Utiliser tsne_hdbscan_labels",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Repertoire de sortie pour les PNG (defaut: meme repertoire que le HDF5)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=True,
        help="Afficher log10(I) (defaut)",
    )
    parser.add_argument(
        "--no-log",
        dest="log",
        action="store_false",
        help="Afficher I directement",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="tab10",
        help="Colormap pour les courbes moyennes par cluster",
    )
    parser.add_argument(
        "--max-curves",
        type=int,
        default=None,
        help="Limiter le trace des courbes individuelles par cluster",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine pour le sous-echantillonnage des courbes individuelles",
    )
    return parser.parse_args()


def protein_name(h5_path: str) -> str:
    stem = Path(h5_path).stem
    token = stem.split("_")[0] if "_" in stem else stem
    return token.strip()


def cluster_output_name(protein: str, label: int) -> str:
    suffix = "noise" if label == -1 else str(label)
    return f"{protein}_cluster_{suffix}.png"


def label_name(args: argparse.Namespace) -> str:
    if args.tsne:
        return "tsne_hdbscan_labels"
    if args.umap:
        return "umap_hdbscan_labels"
    raise ValueError("Choisir --umap ou --tsne pour selectionner les labels de clustering.")


def reconstruct_logI_from_coefs(coefs_row: np.ndarray, n_basis: int, Q: np.ndarray) -> np.ndarray:
    basis = BSplineBasis(
        domain_range=(float(Q.min()), float(Q.max())),
        n_basis=int(n_basis),
    )
    coefs_double = np.asarray(coefs_row, dtype=np.float64).reshape(1, -1)
    fd = FDataBasis(basis, coefs_double)
    return fd.to_grid(grid_points=Q).data_matrix[..., 0].flatten()


def maybe_sample(indices: np.ndarray, max_curves: Optional[int], seed: int) -> np.ndarray:
    if max_curves is None or len(indices) <= max_curves:
        return indices
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=max_curves, replace=False)


def main() -> int:
    args = parse_args()
    ds_label = label_name(args)

    if not Path(args.h5_path).exists():
        raise FileNotFoundError(f"Fichier HDF5 introuvable: {args.h5_path}")

    with h5py.File(args.h5_path, "r") as f:
        for key in REQUIRED:
            if key not in f:
                raise KeyError(f"Dataset obligatoire absent: {key}")

        Q = np.asarray(f["Q"])
        labels = np.asarray(f[ds_label]).astype(int)
        unique_labels = sorted(set(labels.tolist()))

        has_coefs = "bspline_coefs" in f
        has_I = "I" in f
        if not has_coefs and not has_I:
            raise KeyError("Ni 'bspline_coefs' ni 'I' presents dans le fichier.")

        n_basis = None
        if has_coefs and "bspline_nbasis" in f:
            n_basis = int(f["bspline_nbasis"][()])

        # Regrouper les indices par cluster hors de la boucle de tracage
        cluster_indices = {}
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            cluster_indices[lbl] = maybe_sample(idx, args.max_curves, args.seed)

        cmap = plt.colormaps.get_cmap(args.cmap)
        color = {lbl: cmap(i % max(1, cmap.N)) for i, lbl in enumerate(unique_labels)}

        protein = protein_name(args.h5_path)
        out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.h5_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        for plot_idx, lbl in enumerate(unique_labels):
            indices = cluster_indices[lbl]

            log_curves = []
            with h5py.File(args.h5_path, "r") as f:
                if has_coefs and n_basis is not None:
                    coefs_block = np.asarray(f["bspline_coefs"][indices])
                    for row in coefs_block:
                        log_curves.append(reconstruct_logI_from_coefs(row, n_basis, Q))
                else:
                    raw = np.asarray(f["I"][indices], dtype=np.float64)
                    raw = np.maximum(raw, 1e-30)
                    log_curves = [np.log10(row) for row in raw]

            log_curves = np.asarray(log_curves, dtype=np.float64)
            mean_curve = log_curves.mean(axis=0)

            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_axes((0.08, 0.13, 0.90, 0.78))

            for row in log_curves:
                ax.plot(Q, row, color="#aaaaaa", linewidth=0.6, alpha=0.4)

            ax.plot(
                Q,
                mean_curve,
                color=color[lbl],
                linewidth=2.5,
                label=f"Moyenne (n={len(log_curves)})",
            )

            title = f"Cluster {lbl} (n={len(indices)})" if lbl != -1 else f"Noise (n={len(indices)})"
            ax.set_title(title)
            ax.set_xlabel("Q")
            ax.set_ylabel("log10(I)" if args.log else "I")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=9)

            out_file = out_dir / cluster_output_name(protein, lbl)
            fig.savefig(out_file, dpi=200)
            plt.close(fig)
            print(f"[OK] {out_file}")

        print(f"[DONE] {len(unique_labels)} fichiers dans {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
