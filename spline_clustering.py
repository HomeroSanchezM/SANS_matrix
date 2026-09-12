import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from cuml.manifold import UMAP
import hdbscan
import matplotlib.colors as mcolors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(SCRIPT_DIR, "coef.h5")

def _read_vector_from_h5(handle, key):
	if key in handle:
		return np.asarray(handle[key])
	raise KeyError(f"'{key}' not found as dataset or attribute in {H5_PATH}")


def _read_projection_if_valid(handle, key, n_rows, n_cols):
	if key not in handle:
		return None
	arr = np.asarray(handle[key], dtype=np.float64)
	if arr.ndim != 2 or arr.shape != (n_rows, n_cols):
		print(f"Existing '{key}' has shape {arr.shape}, expected ({n_rows}, {n_cols}); recomputing...")
		return None
	return arr


def _plot_embedding(
	embedding_2d,
	color_values,
	title,
	cbar_label,
	out_file,
	categorical=False,
	xlabel="Dim 1",
	ylabel="Dim 2",
	hide_ticks=False,
	noise_alpha=0.05,
	cluster_alpha=0.6,
	highlight_mask=None,   # boolean array, same length as embedding_2d
	legend_loc="upper right",
	legend_fontsize=6,
	legend_title_fontsize=7,
	cmap="RdYlGn",
):
	plt.figure(figsize=(7, 6))

	has_hl = highlight_mask is not None and highlight_mask.any()
	bg_mask = ~highlight_mask if has_hl else np.ones(len(embedding_2d), dtype=bool)

	if categorical:
		# shift labels so -1 becomes 0, others shift by +1
		labels = color_values.copy()
		labels_shifted = labels + 1  # now: -1 → 0, 0 → 1, 1 → 2, ...

		# build colormap: first color = gray, rest = tab20
		base_cmap = mpl.colormaps["tab20"]
		color_list = ["lightgray"] + [base_cmap(i) for i in range(base_cmap.N)]
		custom_cmap = mcolors.ListedColormap(color_list)

		# Build per-point RGBA: noise (label -1) gets noise_alpha, clusters get cluster_alpha
		n_colors = len(color_list)
		norm = mcolors.BoundaryNorm(np.arange(-0.5, n_colors + 0.5), n_colors)
		rgba_colors = custom_cmap(norm(labels_shifted))  # shape (N, 4)
		is_noise = labels == -1
		rgba_colors[is_noise, 3] = noise_alpha
		rgba_colors[~is_noise, 3] = cluster_alpha

		# Background points (all, or non-highlighted)
		plt.scatter(
			embedding_2d[bg_mask, 0],
			embedding_2d[bg_mask, 1],
			c=rgba_colors[bg_mask],
			s=4,
		)
		# Highlighted points on top — larger, with black edge
		if has_hl:
			hl_rgba = rgba_colors[highlight_mask].copy()
			hl_rgba[:, 3] = cluster_alpha  # force full opacity for highlighted
			plt.scatter(
				embedding_2d[highlight_mask, 0],
				embedding_2d[highlight_mask, 1],
				c=hl_rgba,
				s=40,
				edgecolors="black",
				linewidths=0.6,
				zorder=5,
			)

		# Discrete legend: one entry per unique cluster label, sorted
		unique_labels = np.unique(labels)
		legend_handles = []
		for lbl in unique_labels:
			idx = int(lbl) + 1  # shifted index into color_list
			rgba = list(custom_cmap(norm(idx)))
			rgba[3] = noise_alpha if lbl == -1 else cluster_alpha
			patch = mpl.patches.Patch(color=rgba, label="noise" if lbl == -1 else f"cluster {int(lbl)}")
			legend_handles.append(patch)
		if has_hl:
			legend_handles.append(mpl.lines.Line2D(
				[0], [0], marker="o", color="w", markerfacecolor="none",
				markeredgecolor="black", markersize=7, label="highlighted pattern",
			))
		plt.legend(
			handles=legend_handles,
			title=cbar_label,
			fontsize=legend_fontsize,
			title_fontsize=legend_title_fontsize,
			loc=legend_loc,
			markerscale=1.5,
			framealpha=0.6,
		)
	else:
		vmin, vmax = color_values.min(), color_values.max()
		scatter = plt.scatter(
			embedding_2d[bg_mask, 0],
			embedding_2d[bg_mask, 1],
			c=color_values[bg_mask],
			s=1,
			cmap=cmap,
			alpha=0.4,
			edgecolors="none",
			vmin=vmin,
			vmax=vmax,
		)
		# Highlighted points on top
		if has_hl:
			plt.scatter(
				embedding_2d[highlight_mask, 0],
				embedding_2d[highlight_mask, 1],
				c=color_values[highlight_mask],
				s=40,
				cmap=cmap,
				alpha=0.9,
				edgecolors="black",
				linewidths=0.6,
				vmin=vmin,
				vmax=vmax,
				zorder=5,
			)
		cbar = plt.colorbar(scatter, orientation="horizontal", pad=0.1, shrink=0.8)
		cbar.set_label(cbar_label)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if hide_ticks:
		plt.xticks([])
		plt.yticks([])
	plt.tight_layout()
	plt.savefig(out_file, dpi=200)
	plt.close()


def _parse_args():
	parser = argparse.ArgumentParser(description="Compute selected projections and HDBSCAN clustering for spline coefficients")
	parser.add_argument(
		"h5_path",
		nargs="?",
		default="coefs.h5",
		help="HDF5 file name (in same folder as this script) or full path",
	)
	parser.add_argument("--pca", action="store_true", help="Run PCA projection")
	parser.add_argument("--t-sne", "--tsne", dest="tsne", action="store_true", help="Run t-SNE projection")
	parser.add_argument(
		"--umap",
		action="store_true",
		help="Run UMAP projection (2D only, no clustering)",
	)
	parser.add_argument(
		"--umap-clusters",
		action="store_true",
		help="Run UMAP 2D + 5D HDBSCAN clustering",
	)
	parser.add_argument("--all", action="store_true", help="Run all projections (PCA, t-SNE, UMAP, UMAP-clusters)")
	parser.add_argument("--force", action="store_true", help="Force recomputation of t-SNE/UMAP projections even if cached datasets exist")
	parser.add_argument(
		"--hdbscan-min-cluster-size",
		type=int,
		default=5,
		help="HDBSCAN min_cluster_size (smaller -> more/smaller clusters)",
	)
	parser.add_argument(
		"--hdbscan-min-samples",
		type=int,
		default=3,
		help="HDBSCAN min_samples (smaller -> fewer points marked as noise)",
	)
	parser.add_argument(
		"--hdbscan-cluster-selection-epsilon",
		type=float,
		default=0.0,
		help="HDBSCAN cluster_selection_epsilon (smaller -> less merging, usually more clusters)",
	)
	parser.add_argument(
		"--pattern",
		default=None,
		help="Highlight points matching this deuteration pattern (e.g. 00000000000 or 11111111111)",
	)
	parser.add_argument(
		"--hdbscan-cluster-selection-method",
		choices=["eom", "leaf"],
		default="leaf",
		help="HDBSCAN cluster_selection_method ('leaf' usually yields more clusters)",
	)
	parser.add_argument(
		"--color",
		default="d2o_pct",
		help="Scalar dataset name used for point coloring (default: d2o_pct)",
	)
	parser.add_argument(
		"--cmap",
		default="RdYlGn",
		help="Matplotlib colormap name for continuous color scales (default: RdYlGn)",
	)
	return parser.parse_args()


def main():
	global H5_PATH
	args = _parse_args()
	if os.path.isabs(args.h5_path):
		H5_PATH = args.h5_path
	else:
		H5_PATH = os.path.join(SCRIPT_DIR, args.h5_path)

	if not os.path.isfile(H5_PATH):
		fallback_names = ["coefs.h5", "coefs_dataset.h5"]
		for name in fallback_names:
			candidate = os.path.join(SCRIPT_DIR, name)
			if os.path.isfile(candidate):
				print(f"Requested HDF5 not found. Using fallback: {candidate}")
				H5_PATH = candidate
				break
		else:
			raise FileNotFoundError(f"HDF5 file not found: {H5_PATH}")

	if args.hdbscan_min_cluster_size < 2:
		raise ValueError("--hdbscan-min-cluster-size must be >= 2")
	if args.hdbscan_min_samples < 1:
		raise ValueError("--hdbscan-min-samples must be >= 1")
	if args.hdbscan_cluster_selection_epsilon < 0:
		raise ValueError("--hdbscan-cluster-selection-epsilon must be >= 0")

	any_selected = args.pca or args.tsne or args.umap or args.umap_clusters or args.all
	run_pca = args.all or args.pca or not any_selected
	run_tsne = args.all or args.tsne or not any_selected
	run_umap_2d = args.all or args.umap or args.umap_clusters or not any_selected
	run_umap_clusters = args.all or args.umap_clusters

	out_dir = os.path.dirname(H5_PATH)
	print("Loading data from HDF5...")
	with h5py.File(H5_PATH, "r+") as f:
		if "bspline_coefs" not in f:
			raise KeyError("'bspline_coefs' dataset not found in coefs_dataset.h5")

		X = np.asarray(f["bspline_coefs"], dtype=np.float64)
		if X.ndim != 2 or X.shape[1] != 18:
			raise ValueError("'bspline_coefs' must be a 2D matrix with 18 columns")

		if args.color not in f:
			avail = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
			raise KeyError(
				f"Color field '{args.color}' not found in HDF5. "
				f"Available scalar datasets: {avail}"
			)

		color_values = _read_vector_from_h5(f, args.color).reshape(-1)
		if color_values.shape[0] != X.shape[0]:
			raise ValueError(
				f"Color field '{args.color}' length must match bspline_coefs rows"
			)

		# Always load patterns for the highlight feature
		patterns_raw = np.asarray(f["patterns"])
		patterns_str = np.array([
			p.decode() if isinstance(p, (bytes, np.bytes_)) else str(p)
			for p in patterns_raw
		])

		highlight_mask = None
		if args.pattern is not None:
			query = args.pattern.strip("b'\"")
			highlight_mask = patterns_str == query
			n_match = int(highlight_mask.sum())
			if n_match == 0:
				print(f"Warning: pattern '{query}' not found — no points highlighted.")
			else:
				print(f"Pattern '{query}': {n_match} matching points will be highlighted.")

		pca_2d = None
		tsne_2d = None
		tsne_5d = None
		tsne_labels = None
		umap_2d = None
		umap_5d = None
		umap_labels = None
		tsne_2d_new = False
		tsne_5d_new = False
		umap_2d_new = False
		umap_5d_new = False

		# 2D reductions for visualization
		if run_pca:
			print("Computing PCA (2D)...")
			pca_2d = PCA(n_components=2, random_state=42).fit_transform(X)
			print(f"Saving PCA plot colored by {args.color}...")
			_plot_embedding(
				pca_2d,
				color_values,
				f"PCA (2D) colored by {args.color}",
				args.color,
				os.path.join(out_dir, f"pca_2d_{args.color}.png"),
				xlabel="PC1",
				ylabel="PC2",
				cmap=args.cmap,
				highlight_mask=highlight_mask,
			)

		if run_tsne:
			if not args.force:
				tsne_2d = _read_projection_if_valid(f, "tsne_2d", X.shape[0], 2)
			if tsne_2d is None:
				print("Computing t-SNE (2D)...")
				tsne_2d = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(X)
				tsne_2d_new = True
			else:
				print("Using cached t-SNE (2D) from HDF5...")
			print(f"Saving t-SNE plot colored by {args.color}...")
			_plot_embedding(
				tsne_2d,
				color_values,
				f"t-SNE (2D) colored by {args.color}",
				args.color,
				os.path.join(out_dir, f"tsne_2d_{args.color}.png"),
				xlabel="t-SNE 1",
				ylabel="t-SNE 2",
				hide_ticks=True,
				cmap=args.cmap,
				highlight_mask=highlight_mask,
			)

			# 5D t-SNE + HDBSCAN clustering
			if not args.force:
				tsne_5d = _read_projection_if_valid(f, "tsne_5d", X.shape[0], 5)
			if tsne_5d is None:
				print("Computing t-SNE (5D) for clustering...")
				tsne_5d = TSNE(n_components=5, random_state=42, init="pca", learning_rate="auto").fit_transform(X)
				tsne_5d_new = True
			else:
				print("Using cached t-SNE (5D) from HDF5...")
			print("Running HDBSCAN on t-SNE (5D)...")
			tsne_labels = hdbscan.HDBSCAN(
				min_cluster_size=args.hdbscan_min_cluster_size,
				min_samples=args.hdbscan_min_samples,
				cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
				cluster_selection_method=args.hdbscan_cluster_selection_method,
			).fit_predict(tsne_5d)
			print("Saving t-SNE cluster visualization plot...")
			_plot_embedding(
				tsne_2d,
				tsne_labels,
				f"t-SNE (2D) colored by HDBSCAN clusters (5D t-SNE)",
				"cluster label",
				os.path.join(out_dir, "tsne_2d_clusters.png"),
				categorical=True,
				xlabel="t-SNE 1",
				ylabel="t-SNE 2",
				hide_ticks=True,
				highlight_mask=highlight_mask,
			)

		if run_umap_2d:
			if not args.force:
				umap_2d = _read_projection_if_valid(f, "umap_2d", X.shape[0], 2)
			if umap_2d is None:
				print("Computing UMAP (2D)...")
				umap_2d = UMAP(n_components=2, random_state=42).fit_transform(X)
				umap_2d_new = True
			else:
				print("Using cached UMAP (2D) from HDF5...")
			print(f"Saving UMAP plot colored by {args.color}...")
			_plot_embedding(
				umap_2d,
				color_values,
				f"UMAP (2D) colored by {args.color}",
				args.color,
				os.path.join(out_dir, f"umap_2d_{args.color}.png"),
				xlabel="UMAP 1",
				ylabel="UMAP 2",
				hide_ticks=True,
				cmap=args.cmap,
				highlight_mask=highlight_mask if highlight_mask is not None else None,
			)

		if run_umap_clusters:
			umap_5d = None
			umap_labels = None
			if not args.force:
				umap_5d = _read_projection_if_valid(f, "umap_5d", X.shape[0], 5)
			if umap_5d is None:
				print("Computing UMAP (5D) for clustering...")
				umap_5d = UMAP(n_components=5, random_state=42).fit_transform(X)
				umap_5d_new = True
			else:
				print("Using cached UMAP (5D) from HDF5...")
			print("Running HDBSCAN on UMAP (5D)...")
			umap_labels = hdbscan.HDBSCAN(
				min_cluster_size=args.hdbscan_min_cluster_size,
				min_samples=args.hdbscan_min_samples,
				cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
				cluster_selection_method=args.hdbscan_cluster_selection_method,
			).fit_predict(umap_5d)
			print("Saving UMAP cluster visualization plot...")
			_plot_embedding(
				umap_2d if umap_2d is not None else UMAP(n_components=2, random_state=42).fit_transform(X),
				umap_labels,
				"UMAP (2D) colored by HDBSCAN clusters (5D UMAP)",
				"cluster label",
				os.path.join(out_dir, "umap_2d_clusters.png"),
				categorical=True,
				xlabel="UMAP 1",
				ylabel="UMAP 2",
				hide_ticks=True,
				highlight_mask=highlight_mask,
				legend_loc="lower right",
				legend_fontsize=8,
				legend_title_fontsize=9,
			)

		# Store cluster label vectors as datasets (attributes can fail for large vectors)
		print("Writing selected outputs to HDF5...")
		if tsne_labels is not None:
			if "tsne_hdbscan_labels" in f:
				del f["tsne_hdbscan_labels"]
			f.create_dataset("tsne_hdbscan_labels", data=tsne_labels.astype(np.int32))
			
		if umap_labels is not None:
			if "umap_hdbscan_labels" in f:
				del f["umap_hdbscan_labels"]
			f.create_dataset("umap_hdbscan_labels", data=umap_labels.astype(np.int32))

		# Store selected projections as datasets (attributes can't hold large arrays reliably)
		datasets_to_write = []
		if pca_2d is not None:
			datasets_to_write.append(("pca_2d", pca_2d))
		if tsne_2d_new:
			datasets_to_write.append(("tsne_2d", tsne_2d))
		if umap_2d_new:
			datasets_to_write.append(("umap_2d", umap_2d))
		if tsne_5d_new:
			datasets_to_write.append(("tsne_5d", tsne_5d))
		if umap_5d_new:
			datasets_to_write.append(("umap_5d", umap_5d))

		for ds_name, data in datasets_to_write:
			if ds_name in f:
				del f[ds_name]
			f.create_dataset(ds_name, data=data.astype(np.float32))
		print("Done.")


if __name__ == "__main__":
	main()
