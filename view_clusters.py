#!/usr/bin/env python3
"""
Visualize curves reconstructed from B‑spline coefficients.
Usage:
    python view_clusters.py coefs.h5 --tsne
    python view_clusters.py coefs.h5 --umap --max-curves 300 --save fig.png
    python view_clusters.py coefs.h5 --tsne --info
"""

import argparse
import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from skfda.representation.basis import BSplineBasis
import skfda


def load_compact_h5(path: Path, cluster_method: str = None):
    """Load compact .h5. If cluster_method is 'tsne' or 'umap', also loads cluster labels."""
    with h5py.File(path, 'r') as f:
        patterns  = f['patterns'][:].astype(str)
        d2o_pct   = f['d2o_pct'][:]
        ratio     = f['ratio'][:]     if 'ratio'   in f else None
        fitness   = f['fitness'][:]   if 'fitness' in f else None
        Q         = f['Q'][:]
        coefs     = f['bspline_coefs'][:]
        n_basis   = int(f['bspline_nbasis'][()]) if 'bspline_nbasis' in f else coefs.shape[1]
        print(n_basis)
        labels = None
        if cluster_method is not None:
            label_key = 'tsne_hdbscan_labels' if cluster_method == 'tsne' else 'umap_hdbscan_labels'
            if label_key not in f:
                sys.exit(f"Label key '{label_key}' not found in {path}.")
            labels = f[label_key][:]
    return Q, coefs, n_basis, patterns, d2o_pct, fitness, ratio, labels


def reconstruct_log_I(Q, coefs, n_basis):
    """Reconstruct log(I) by evaluating the B‑spline basis with the stored coefficients."""
    basis = BSplineBasis(
        domain_range=(float(Q.min()), float(Q.max())),
        n_basis=n_basis,
    )
    fd_basis = skfda.FDataBasis(basis=basis, coefficients=coefs)
    log_I = fd_basis.to_grid(grid_points=Q).data_matrix[..., 0]
    return log_I.astype(np.float32)


def plot_clusters(
    Q, log_I, d2o_pct, labels,
    method, max_curves, seed, save,
):
    """Plot one panel per cluster group, curves colored by %D2O."""
    unique_labels = np.unique(labels)
    cluster_labels = unique_labels[unique_labels >= 0]
    noise_labels   = unique_labels[unique_labels < 0]
    all_panels     = list(cluster_labels) + list(noise_labels)

    n_panels = len(all_panels)
    if n_panels == 0:
        sys.exit("No clusters found.")

    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    norm = mcolors.Normalize(vmin=float(d2o_pct.min()), vmax=float(d2o_pct.max()))
    cmap = plt.colormaps['coolwarm']

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    rng = np.random.default_rng(seed)

    # Common y-axis limits across all panels
    y_min = float(np.quantile(log_I, 0.005))
    y_max = float(np.quantile(log_I, 0.995))

    for panel_idx, lbl in enumerate(all_panels):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        mask = labels == lbl
        lI   = log_I[mask]
        d2o  = d2o_pct[mask]

        n_total = len(lI)
        if n_total > max_curves:
            idx = rng.choice(n_total, size=max_curves, replace=False)
            idx.sort()
            lI, d2o = lI[idx], d2o[idx]
            note = f"{n_total} curve{'s' if n_total != 1 else ''} (sampled)"
        else:
            note = f"{n_total} curve{'s' if n_total != 1 else ''}"

        for i in range(len(lI)):
            ax.plot(Q, lI[i], color=cmap(norm(float(d2o[i]))),
                    alpha=0.4, linewidth=0.8)

        panel_title = f"Noise ({note})" if lbl < 0 else f"Cluster {lbl}  ({note})"
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel('q  (Å⁻¹)', fontsize=9)
        ax.set_ylabel('log I(q)', fontsize=9)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    # Hide unused axes
    for panel_idx in range(n_panels, nrows * ncols):
        axes[panel_idx // ncols][panel_idx % ncols].set_visible(False)

    method_label = 't-SNE' if method == 'tsne' else 'UMAP'
    fig.suptitle(f'Clustered SANS curves — {method_label} + HDBSCAN\n(colored by %D\u2082O)',
                 fontsize=13, y=1.01)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Reserve space at bottom and place colorbar below panel grid
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation='horizontal',
        fraction=0.05,
        pad=0.08,
        label='%D\u2082O',
    )

    out_path = save if save else "clusters.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {out_path}")
    plt.close(fig)


def plot_curves(
    Q, log_I, patterns, d2o_pct, fitness, ratio,
    max_curves, color_by, pattern_filter, seed, save,
    highlight_best=False,
):
    # Filter by pattern
    if pattern_filter is not None:
        mask = patterns == pattern_filter
        if not mask.any():
            sys.exit(f"Pattern '{pattern_filter}' not found.")
        log_I, patterns, d2o_pct = log_I[mask], patterns[mask], d2o_pct[mask]
        if fitness is not None:
            fitness, ratio = fitness[mask], ratio[mask]

    n_total = len(log_I)

    # plot curves (random sample)
    rng = np.random.default_rng(seed)
    if n_total > max_curves:
        idx = rng.choice(n_total, size=max_curves, replace=False)
        idx.sort()
        log_I, patterns, d2o_pct = log_I[idx], patterns[idx], d2o_pct[idx]
        if fitness is not None:
            fitness, ratio = fitness[idx], ratio[idx]
        sample_note = f"sample {max_curves}/{n_total}"
    else:
        sample_note = f"{n_total} curve{'s' if n_total > 1 else ''}"

    # Choose colormap
    if color_by == 'fitness' and fitness is not None:
        color_vals = fitness
        cbar_label = 'fitness'
        cmap_name  = 'plasma'
    elif color_by == 'ratio' and ratio is not None:
        color_vals = ratio
        cbar_label = 'ratio'
        cmap_name  = 'viridis'
    else:   # default: %D₂O
        color_vals = d2o_pct.astype(float)
        cbar_label = '%D\u2082O'
        cmap_name  = 'coolwarm'

    norm = mcolors.Normalize(vmin=color_vals.min(), vmax=color_vals.max())
    cmap = plt.colormaps[cmap_name]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all curves
    for i in range(len(log_I)):
        ax.plot(Q, log_I[i], color=cmap(norm(color_vals[i])),
                alpha=0.35, linewidth=0.8)

    # Highlight best curve 
    best_idx = None
    if highlight_best and fitness is not None and len(fitness) > 0:
        valid_fit = ~np.isnan(fitness)
        if not np.any(valid_fit):
            print("Warning: all fitness values are NaN; cannot highlight best curve.")
        else:
            best_idx = np.nanargmax(fitness)
            I_best = log_I[best_idx]                   
            d2o_best = d2o_pct[best_idx]
            pattern_best = patterns[best_idx]
            fitness_best = fitness[best_idx]
            ratio_best = ratio[best_idx] if ratio is not None else None

            # Plot the best curve prominently
            ax.plot(Q, I_best, color='black', linewidth=2.5, zorder=10,
                    label='Best fitness')

            # Annotation with metrics (same style as view_big_matrix)
            textstr = (f"Fitness: {fitness_best:.8f}\n"
                       f"Ratio: {ratio_best:.3f}\n"
                       f"Pattern: {pattern_best}\n"
                       f"D\u2082O: {d2o_best}%")
            mid_q = len(Q) // 2
            x_text, y_text = Q[mid_q], I_best[mid_q]
            ax.annotate(
                textstr,
                xy=(x_text, y_text),
                xytext=(30, 30),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          alpha=0.9, edgecolor='gray'),
                arrowprops=dict(arrowstyle='->', color='gray'),
                zorder=20,
            )
            ax.legend(fontsize=9, loc='lower center', bbox_to_anchor=(0.5, -0.12))

    ax.set_xlabel('q  (Å⁻¹)', fontsize=12)
    ax.set_ylabel('log I(q)', fontsize=12)

    title = f'Reconstructed SANS curves from B‑spline  ({sample_note})'
    if pattern_filter:
        title += f'\npattern = {pattern_filter}'
    if highlight_best and best_idx is not None:
        title += '\nbest curve in bold'
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label, fontsize=11)

    if color_by == 'd2o':
        uniq = np.unique(d2o_pct)
        cbar.set_ticks(uniq[::max(1, len(uniq) // 10)])

    plt.tight_layout()

    out_path = save if save else "curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize curves in cluster mode (t-SNE/UMAP + HDBSCAN)."
    )
    parser.add_argument("h5", help="Compact .h5 file")
    parser.add_argument("--max-curves", type=int, default=200,
                        help="Max curves to show (default 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default=None,
                        help="Save figure (e.g. fig.png)")
    parser.add_argument("--info", action="store_true",
                        help="Print file statistics only, no plot")
    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument("--tsne", action="store_true",
                              help="Cluster mode: plot one panel per t-SNE + HDBSCAN cluster")
    method_group.add_argument("--umap", action="store_true",
                              help="Cluster mode: plot one panel per UMAP + HDBSCAN cluster")
    args = parser.parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        sys.exit(f"File not found: {h5_path}")

    cluster_method = 'tsne' if args.tsne else 'umap'

    save_path = Path(args.save) if args.save else h5_path.with_name(f"{h5_path.stem}_{cluster_method}_clusters.png")
    if save_path.suffix.lower() != '.png':
        save_path = save_path.with_suffix('.png')

    print(f"\nLoading {h5_path} ...")
    Q, coefs, n_basis, patterns, d2o_pct, fitness, ratio, labels = load_compact_h5(h5_path, cluster_method)
    n_curves, n_b = coefs.shape

    print(f"  Curves          : {n_curves}")
    print(f"  B‑spline bases  : {n_b}  (n_basis={n_basis})")
    print(f"  Q               : [{Q.min():.4f}, {Q.max():.4f}] Å⁻¹  ({len(Q)} points)")
    print(f"  Unique patterns : {len(np.unique(patterns))}")
    print(f"  Unique D₂O      : {np.unique(d2o_pct).tolist()} %")
    if fitness is not None:
        print(f"  Fitness         : [{np.nanmin(fitness):.3e}, {np.nanmax(fitness):.3e}]")
    if ratio is not None:
        print(f"  Ratio           : [{np.nanmin(ratio):.3f}, {np.nanmax(ratio):.3f}]")

    if args.info:
        return

    print("\nReconstructing log(I) from coefficients ...")
    log_I = reconstruct_log_I(Q, coefs, n_basis)
    print(f"  log(I) range    : [{log_I.min():.3f}, {log_I.max():.3f}]")

    method_label = 't-SNE' if cluster_method == 'tsne' else 'UMAP'
    unique_labels = np.unique(labels)
    n_clusters = int((unique_labels >= 0).sum())
    n_noise    = int((labels < 0).sum())
    print(f"  Clusters        : {n_clusters}  (noise points: {n_noise})")
    print(f"\nGenerating cluster plot ({method_label} + HDBSCAN) ...")
    plot_clusters(
        Q, log_I, d2o_pct, labels,
        method=cluster_method,
        max_curves=args.max_curves,
        seed=args.seed,
        save=str(save_path),
    )

if __name__ == "__main__":
    main()
