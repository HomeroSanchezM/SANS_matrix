#!/usr/bin/env python3
"""
bspline_fitting.py — Ajuste B-spline sobre log(I) de las curvas SANS.

Lee el HDF5 generado por main_matrix.py, ajusta B-splines sobre log(I(q))
y guarda los coeficientes en el mismo fichero.  Genera también un plot de
diagnóstico para verificar la calidad del ajuste.

Estructura HDF5 de entrada (acceso por nombre o índice):
  '0'  / 'patterns'           (n_curves,)           str
  '1'  / 'd2o_pct'            (n_curves,)           int32
  '2'  / 'ratio'              (n_curves,)           f32
  '3'  / 'product_areas'      (n_curves,)           f32
  '4'  / 'Q'                  (n_points,)           f32
  '5'  / 'I'                  (n_curves, n_points)  f32
  '6'  / 'deuterium_pct'      (n_curves,)           f32   ← %D total
  '9'  / 'nonlabile_deut_pct' (n_curves,)           f32   ← %D no lábil
  '10' / 'rg'                 (n_curves,)           f32   ← radio de giro (Å)

Datasets añadidos al HDF5 original:
  'bspline_coefs'   (n_valid, n_basis)  f32
  'bspline_mask'    (n_curves,)         u8
  'bspline_nbasis'  scalar int
  Hard links: '7' → bspline_coefs, '8' → bspline_mask

HDF5 compacto exportado con --export (sin I original):
  '0' / 'patterns'           (n_valid,)           str
  '1' / 'd2o_pct'            (n_valid,)           int32
  '2' / 'ratio'              (n_valid,)           f32
  '3' / 'product_areas'      (n_valid,)           f32
  '4' / 'Q'                  (n_points,)          f32
  '5' / 'bspline_coefs'      (n_valid, n_basis)   f32
  '6' / 'deuterium_pct'      (n_valid,)           f32
  '7' / 'nonlabile_deut_pct' (n_valid,)           f32
  '8' / 'rg'                 (n_valid,)           f32   ← radio de giro (Å)
         'bspline_nbasis'     scalar int
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

import skfda
from skfda.representation.basis import BSplineBasis


# ─────────────────────────────────────────────
# Carga
# ─────────────────────────────────────────────
def load_h5(path: Path):
    """
    Carga Q, I y metadatos desde el HDF5 (acceso por nombre o índice numérico).
    Devuelve también deuterium_pct, nonlabile_deut_pct y rg si existen.
    """
    with h5py.File(path, 'r') as f:
        Q                  = f['Q'][:]
        I                  = f['I'][:]
        patterns           = f['patterns'][:].astype(str)
        d2o_pct            = f['d2o_pct'][:]
        ratio              = f['ratio'][:]               if 'ratio'              in f else None
        product_areas      = f['product_areas'][:]       if 'product_areas'      in f else None
        fitness            = f['fitness'][:]             if 'fitness'            in f else None
        deuterium_pct      = f['deuterium_pct'][:]       if 'deuterium_pct'      in f else None
        nonlabile_deut_pct = f['nonlabile_deut_pct'][:] if 'nonlabile_deut_pct' in f else None
        rg                 = f['rg'][:]                  if 'rg'                 in f else None
    return Q, I, patterns, d2o_pct, ratio, product_areas, fitness, deuterium_pct, nonlabile_deut_pct, rg


# ─────────────────────────────────────────────
# Ajuste B-spline
# ─────────────────────────────────────────────
def fit_bsplines(Q: np.ndarray, I: np.ndarray, n_basis: int):
    """
    Ajusta B-splines sobre log(I) para cada curva con I > 0.

    Retorna:
      coefs      (n_valid, n_basis)  float32
      mask       (n_curves,)         bool
      log_I      (n_valid, n_points) float32
      basis_fd   FDataBasis ajustado
    """
    mask = np.all(I > 0, axis=1)
    n_invalid = (~mask).sum()
    if n_invalid:
        print(f"  Advertencia: {n_invalid} curvas ignoradas (I <= 0).")

    log_I = np.log10(I[mask]).astype(np.float32)

    print(f"  Curvas validas : {mask.sum()} / {len(I)}")
    print(f"  Rango log(I)   : [{log_I.min():.3f}, {log_I.max():.3f}]")

    fd    = skfda.FDataGrid(log_I, grid_points=Q)
    basis = BSplineBasis(
        domain_range=(float(Q.min()), float(Q.max())),
        n_basis=n_basis,
    )

    print(f"  Ajustando {n_basis} bases B-spline en [{Q.min():.4f}, {Q.max():.4f}] A^-1 ...")
    basis_fd = fd.to_basis(basis)

    coefs = basis_fd.coefficients.astype(np.float32)
    print(f"  Coeficientes   : shape {coefs.shape}")

    recon = basis_fd.to_grid(grid_points=Q).data_matrix[..., 0]
    rmse  = float(np.sqrt(np.mean((recon - log_I) ** 2)))
    print(f"  RMSE log(I)    : {rmse:.5f}")

    return coefs, mask, log_I, basis_fd


# ─────────────────────────────────────────────
# Guardado en el HDF5 original
# ─────────────────────────────────────────────
def save_to_h5(path: Path, coefs: np.ndarray, mask: np.ndarray, n_basis: int):
    """
    Añade coeficientes y máscara al HDF5 existente (in-place).
    Hard links: '7' -> bspline_coefs, '8' -> bspline_mask
    """
    with h5py.File(path, 'a') as f:
        for key in ('bspline_coefs', 'bspline_mask', 'bspline_nbasis', '7', '8'):
            if key in f:
                del f[key]

        f.create_dataset('bspline_coefs',   data=coefs,                dtype='<f4')
        f.create_dataset('bspline_mask',    data=mask.astype(np.uint8), dtype='uint8')
        f.create_dataset('bspline_nbasis',  data=np.int32(n_basis))

        f['7'] = f['bspline_coefs']
        f['8'] = f['bspline_mask']

    print(f"  HDF5 actualizado: bspline_coefs {coefs.shape}, bspline_mask {mask.shape}")


# ─────────────────────────────────────────────
# Exportar HDF5 compacto (sin I original)
# ─────────────────────────────────────────────
def export_compact_h5(
    export_path: Path,
    Q: np.ndarray,
    I: np.ndarray,              # intensités originales (n_curves, n_points) f32
    coefs: np.ndarray,
    mask: np.ndarray,
    n_basis: int,
    patterns: np.ndarray,
    d2o_pct: np.ndarray,
    ratio,
    product_areas,
    deuterium_pct,
    nonlabile_deut_pct,
    rg,                         # radio de giro (n_curves,) f32 | None
    ratio_threshold: float = 0.0,
    imax_threshold: float = 0.01,
    fitness=None,
):
    """
    Escribe un HDF5 compacto con todo menos los I originales.
    Solo incluye curvas válidas (mask=True) y, opcionalmente, filtra por ratio.

    Estructura y orden de índices:
      '0' / 'patterns'           (n_valid,)          str
      '1' / 'd2o_pct'            (n_valid,)          int32
      '2' / 'ratio'              (n_valid,)          f32
      '3' / 'product_areas'      (n_valid,)          f32
      '4' / 'Q'                  (n_points,)         f32
      '5' / 'bspline_coefs'      (n_valid, n_basis)  f32
      '6' / 'deuterium_pct'      (n_valid,)          f32
      '7' / 'nonlabile_deut_pct' (n_valid,)          f32
      '8' / 'rg'                 (n_valid,)          f32   ← radio de giro (Å)
             'bspline_nbasis'     scalar int
    """
    # Combinar la máscara de I>0 con el filtro de ratio
    if ratio is not None and ratio_threshold > 0.0:
        ratio_mask    = ratio >= ratio_threshold
        combined_mask = mask & ratio_mask
        n_rejected    = int(mask.sum()) - int(combined_mask.sum())
        print(f"  Ratio threshold  : >= {ratio_threshold}  "
              f"({n_rejected} curvas excluidas por ratio bajo)")
    else:
        combined_mask = mask

    # Filtre sur I_max : exclure les courbes dont le maximum de I es < imax_threshold
    imax_per_curve = I.max(axis=1)                         # (n_curves,)
    imax_mask      = imax_per_curve >= imax_threshold
    n_imax_rej     = int(combined_mask.sum()) - int((combined_mask & imax_mask).sum())
    combined_mask  = combined_mask & imax_mask
    print(f"  I_max threshold  : >= {imax_threshold}  "
          f"(log10 >= {np.log10(imax_threshold):.2f})  "
          f"({n_imax_rej} curvas excluidas por I_max bajo)")

    coefs_export = coefs[combined_mask[mask]]

    n_valid   = int(combined_mask.sum())
    pat_valid = patterns[combined_mask]
    d2o_valid = d2o_pct[combined_mask].astype(np.int32)
    rat_valid = ratio[combined_mask].astype(np.float32)   if ratio   is not None else np.zeros(n_valid, dtype=np.float32)
    pa_valid  = product_areas[combined_mask].astype(np.float32) if product_areas is not None else np.zeros(n_valid, dtype=np.float32)

    # deuterium_pct (total)
    if deuterium_pct is not None:
        deut_valid = deuterium_pct[combined_mask].astype(np.float32)
    else:
        deut_valid = np.full(n_valid, np.nan, dtype=np.float32)
        print("  Avertissement : deuterium_pct absent du HDF5 source → NaN dans l'export.")

    # nonlabile_deut_pct
    if nonlabile_deut_pct is not None:
        nld_valid = nonlabile_deut_pct[combined_mask].astype(np.float32)
    else:
        nld_valid = np.full(n_valid, np.nan, dtype=np.float32)
        print("  Avertissement : nonlabile_deut_pct absent du HDF5 source → NaN dans l'export.")

    # rg
    if rg is not None:
        rg_valid = rg[combined_mask].astype(np.float32)
    else:
        rg_valid = np.full(n_valid, np.nan, dtype=np.float32)
        print("  Avertissement : rg absent du HDF5 source → NaN dans l'export.")

    with h5py.File(export_path, 'w') as f:
        f.create_dataset('patterns',           data=pat_valid.astype(h5py.string_dtype()))
        f.create_dataset('d2o_pct',            data=d2o_valid,      dtype='int32')
        f.create_dataset('ratio',              data=rat_valid,      dtype='<f4')
        f.create_dataset('product_areas',      data=pa_valid,       dtype='<f4')
        f.create_dataset('Q',                  data=Q,              dtype='<f4')
        f.create_dataset('bspline_coefs',      data=coefs_export,   dtype='<f4')
        f.create_dataset('deuterium_pct',      data=deut_valid,     dtype='<f4')
        f.create_dataset('nonlabile_deut_pct', data=nld_valid,      dtype='<f4')
        f.create_dataset('rg',                 data=rg_valid,       dtype='<f4')
        f.create_dataset('bspline_nbasis',     data=np.int32(n_basis))

        # Hard links:
        #   0=patterns, 1=d2o_pct, 2=ratio, 3=product_areas,
        #   4=Q,        5=bspline_coefs, 6=deuterium_pct,
        #   7=nonlabile_deut_pct,         8=rg
        for idx_str, name in {
            '0': 'patterns',           '1': 'd2o_pct',
            '2': 'ratio',              '3': 'product_areas',
            '4': 'Q',                  '5': 'bspline_coefs',
            '6': 'deuterium_pct',      '7': 'nonlabile_deut_pct',
            '8': 'rg',
        }.items():
            f[idx_str] = f[name]

    size_mb = export_path.stat().st_size / 1e6
    print(f"  Fichero compacto : {export_path}  ({size_mb:.2f} MB)")
    print(f"  Contenido        : {n_valid} curvas válidas x {n_basis} coefs B-spline")
    print(f"  Acceso           : f['bspline_coefs'] o f['5']")
    if not np.all(np.isnan(rg_valid)):
        print(f"  Rg (Å)           : [{np.nanmin(rg_valid):.3f}, {np.nanmax(rg_valid):.3f}]")
    if nld_valid is not None:
        print(f"  %D no lábil      : [{np.nanmin(nld_valid):.2f}, {np.nanmax(nld_valid):.2f}]")


# ─────────────────────────────────────────────
# Plot de diagnóstico
# ─────────────────────────────────────────────
def plot_diagnostics(
    Q: np.ndarray,
    log_I: np.ndarray,
    basis_fd,
    patterns: np.ndarray,
    d2o_pct: np.ndarray,
    deuterium_pct,                # %D total (puede ser None)
    nonlabile_deut_pct,           # %D no lábil (puede ser None)
    rg,                           # radio de giro en Å (puede ser None)
    n_show: int,
    seed: int,
    save,
    worst_n: int | None = None,
):
    recon  = basis_fd.to_grid(grid_points=Q).data_matrix[..., 0]
    n_val  = len(log_I)

    if worst_n is not None:
        per_curve_rmse = np.sqrt(np.mean((recon - log_I) ** 2, axis=1))
        n_select = min(worst_n, n_val)
        idx = np.argsort(per_curve_rmse)[::-1][:n_select]
        title_mode = f"Top-{n_select} curvas con peor RMSE"
        n_show = n_select
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_val, size=min(n_show, n_val), replace=False)
        idx.sort()
        title_mode = f"{len(idx)} curvas aleatorias"

    n_cols = min(3, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols
    colors = plt.cm.tab10.colors

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(
        f"Diagnóstico B-spline  |  {basis_fd.basis.n_basis} bases  |  "
        f"{n_val} curvas válidas  |  {title_mode}\n"
        f"Eje Y = log(I),   Q in [{Q.min():.3f}, {Q.max():.3f}] A^-1",
        fontsize=12, y=1.01,
    )

    for plot_i, curve_i in enumerate(idx):
        ax    = fig.add_subplot(n_rows, n_cols, plot_i + 1)
        orig  = log_I[curve_i]
        fit   = recon[curve_i]
        resid = orig - fit

        ax.plot(Q, orig, color=colors[0], lw=1.5, alpha=0.85, label='log(I) original')
        ax.plot(Q, fit,  color=colors[1], lw=1.5, alpha=0.95,
                linestyle='--', label='B-spline')

        ax2 = ax.twinx()
        ax2.plot(Q, resid, color='gray', lw=0.8, alpha=0.6, linestyle=':')
        ax2.axhline(0, color='gray', lw=0.5, linestyle='--')
        ax2.set_ylabel('residuo', fontsize=7, color='gray')
        ax2.tick_params(axis='y', labelsize=6, colors='gray')
        ax2.set_ylim(-0.5, 0.5)

        rmse_i = float(np.sqrt(np.mean(resid ** 2)))

        # Título enriquecido con %D total, no lábil y Rg
        parts = [f"pat={patterns[curve_i]}", f"D2O={int(d2o_pct[curve_i])}%"]
        if deuterium_pct is not None:
            parts.append(f"%D={deuterium_pct[curve_i]:.1f}%")
        if nonlabile_deut_pct is not None:
            parts.append(f"%DNL={nonlabile_deut_pct[curve_i]:.1f}%")
        if rg is not None and not np.isnan(rg[curve_i]):
            parts.append(f"Rg={rg[curve_i]:.2f}Å")
        ax.set_title("  ".join(parts) + f"\nRMSE={rmse_i:.4f}", fontsize=8)

        ax.set_xlabel('q  (A^-1)', fontsize=8)
        ax.set_ylabel('log(I)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
        if plot_i == 0:
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  Figura guardada: {save}")
    else:
        plt.show()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Ajuste B-spline sobre log(I) de las curvas SANS."
    )
    parser.add_argument("h5",           help="Fichero HDF5 de entrada (output.h5)")
    parser.add_argument("--n-basis",    type=int,  default=10,
                        help="Número de bases B-spline (default 10)")
    parser.add_argument("--plot-n",     type=int,  default=6,
                        help="Curvas en el plot de diagnóstico (default 6)")
    parser.add_argument("--seed",       type=int,  default=42,
                        help="Semilla para selección aleatoria de curvas")
    parser.add_argument("--worst",      type=int,  default=None, metavar="N",
                        help="Mostrar las N curvas con peor RMSE en lugar de "
                             "una selección aleatoria (ej: --worst 12)")
    parser.add_argument("--save",       default=None,
                        help="Guardar plot en lugar de mostrarlo (ej: diag.png)")
    parser.add_argument("--no-save-h5", action="store_true",
                        help="No modificar el HDF5 original, solo generar el plot")
    parser.add_argument("--export",          default=None, metavar="FICHERO.h5",
                        help="Exportar HDF5 compacto sin I original (ej: coefs.h5)")
    parser.add_argument("--ratio-threshold", type=float, default=0.0,
                        metavar="R",
                        help="Excluir del export curvas con ratio < R (default: sin filtro)")
    parser.add_argument("--imax-threshold", type=float, default=0.01,
                        metavar="IMAX",
                        help="Excluir del export curvas con I_max < IMAX "
                             "(default: 0.01, equivale a log10(I_max) < -2)")
    args = parser.parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        sys.exit(f"Fichero no encontrado: {h5_path}")

    print(f"\n-- Cargando {h5_path} --")
    Q, I, patterns, d2o_pct, ratio, product_areas, fitness, deuterium_pct, nonlabile_deut_pct, rg = load_h5(h5_path)
    print(f"  {len(I)} curvas x {len(Q)} puntos Q")
    print(f"  Q : [{Q.min():.4f}, {Q.max():.4f}] A^-1")
    print(f"  I : [{I.min():.3e}, {I.max():.3e}]")
    if ratio is not None:
        print(f"  ratio : [{np.nanmin(ratio):.4f}, {np.nanmax(ratio):.4f}]")
    if product_areas is not None:
        print(f"  product_areas : [{np.nanmin(product_areas):.4e}, {np.nanmax(product_areas):.4e}]")
    if fitness is not None:
        print(f"  fitness : [{np.nanmin(fitness):.4e}, {np.nanmax(fitness):.4e}]")
    else:
        print("  fitness : no disponible (formato nuevo, usar read_matrix.py --gamma)")
    if deuterium_pct is not None:
        print(f"  %D total : [{np.nanmin(deuterium_pct):.2f}, {np.nanmax(deuterium_pct):.2f}] %")
    else:
        print("  %D total : no disponible (formato antiguo)")
    if nonlabile_deut_pct is not None:
        print(f"  %D no lábil : [{np.nanmin(nonlabile_deut_pct):.2f}, {np.nanmax(nonlabile_deut_pct):.2f}] %")
    else:
        print("  %D no lábil : no disponible (formato antiguo)")
    if rg is not None:
        print(f"  Rg (Å) : [{np.nanmin(rg):.3f}, {np.nanmax(rg):.3f}]  "
              f"({np.isnan(rg).sum()} NaN)")
    else:
        print("  Rg : no disponible (formato antiguo o sin .log)")

    print(f"\n-- Ajuste B-spline (n_basis={args.n_basis}) --")
    coefs, mask, log_I, basis_fd = fit_bsplines(Q, I, args.n_basis)

    if not args.no_save_h5:
        print(f"\n-- Guardando coeficientes en {h5_path} --")
        save_to_h5(h5_path, coefs, mask, args.n_basis)

    if args.export:
        print(f"\n-- Exportando HDF5 compacto -> {args.export} --")
        export_compact_h5(
            Path(args.export),
            Q, I, coefs, mask, args.n_basis,
            patterns, d2o_pct, ratio, product_areas,
            deuterium_pct,
            nonlabile_deut_pct,
            rg,
            fitness=fitness,
            ratio_threshold=args.ratio_threshold,
            imax_threshold=args.imax_threshold,
        )

    print(f"\n-- Plot de diagnóstico --")
    # Alinear datos válidos con log_I (mask ya aplicado en fit_bsplines)
    deut_valid = deuterium_pct[mask]      if deuterium_pct      is not None else None
    nld_valid  = nonlabile_deut_pct[mask] if nonlabile_deut_pct is not None else None
    rg_valid   = rg[mask]                 if rg                 is not None else None

    plot_diagnostics(
        Q, log_I, basis_fd,
        patterns[mask], d2o_pct[mask],
        deut_valid,
        nld_valid,
        rg_valid,
        n_show=args.plot_n, seed=args.seed, save=args.save,
        worst_n=args.worst,
    )

    print("\n Listo.")


if __name__ == "__main__":
    main()
