#!/usr/bin/env python3
"""
Fase 1 – Pipeline de generación de curvas SANS
================================================
Genera todas las combinaciones posibles de deuteración de aminoácidos
y porcentajes de D₂O, lanza Pepsi-SANS en paralelo y construye un archivo HDF5
con el vector Q, todas las curvas I(q), los metadatos de patrón y %D₂O,
y les métriques de product_areas (multi-referencia) et ratio Imax/fondo.

Las referencias para el cálculo de product_areas son las propias curvas simuladas del
patrón 0 (todos protonados en posiciones no lábiles) a 0 % y 100 % D₂O.

Nuevos campos en el HDF5:
  'deuterium_pct'       (N_curves,)  f32  — % de deuterio total de la proteína
  'nonlabile_deut_pct'  (N_curves,)  f32  — % de deuterio en posiciones NO lábiles
  'rg'                  (N_curves,)  f32  — radio de giro (Å) extraído del .log de Pepsi-SANS
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gemmi
import h5py
import numpy as np
from scipy.integrate import simpson
from scipy.optimize import curve_fit

# ----------------------------------------------------------------------
# Configuración de logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger("sans_phase1")

# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------
FLOAT_SIZE = 4                       # bytes por f32

# Regex para extraer el Rg del contraste de longitud de dispersión del .log de Pepsi-SANS.
# Captura específicamente la línea:
#   "Radius of gyration (the diff. of sc. length)........ : 22.1182 A"
# La valeur peut être "-nan" (contrast nul) → on la remplacera par 0.0.
_RG_PATTERN = re.compile(
    r'^Radius of gyration \(the diff\. of sc\. length\)\.+\s*:\s*(-?nan|[\d.eE+\-]+)\s*A',
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Definición de grupos de aminoácidos
# ----------------------------------------------------------------------
COMPLETE_GROUPS = [
    {"label": "ALA",     "residues": ["ALA"]},
    {"label": "ARG",     "residues": ["ARG"]},
    {"label": "ASN/ASP", "residues": ["ASN", "ASP"]},
    {"label": "CYS",     "residues": ["CYS"]},
    {"label": "GLU/GLN", "residues": ["GLU", "GLN"]},
    {"label": "GLY",     "residues": ["GLY"]},
    {"label": "HIS",     "residues": ["HIS"]},
    {"label": "ILE",     "residues": ["ILE"]},
    {"label": "LEU",     "residues": ["LEU"]},
    {"label": "LYS",     "residues": ["LYS"]},
    {"label": "MET",     "residues": ["MET"]},
    {"label": "PHE",     "residues": ["PHE"]},
    {"label": "PRO",     "residues": ["PRO"]},
    {"label": "SER",     "residues": ["SER"]},
    {"label": "THR",     "residues": ["THR"]},
    {"label": "TRP",     "residues": ["TRP"]},
    {"label": "TYR",     "residues": ["TYR"]},
    {"label": "VAL",     "residues": ["VAL"]},
]

REDUCED_GROUPS = [
    {"label": "ARG",     "residues": ["ARG"]},
    {"label": "GLU/GLN", "residues": ["GLU", "GLN"]},
    {"label": "HIS",     "residues": ["HIS"]},
    {"label": "ILE",     "residues": ["ILE"]},
    {"label": "LEU",     "residues": ["LEU"]},
    {"label": "LYS",     "residues": ["LYS"]},
    {"label": "PHE",     "residues": ["PHE"]},
    {"label": "PRO",     "residues": ["PRO"]},
    {"label": "THR",     "residues": ["THR"]},
    {"label": "TYR",     "residues": ["TYR"]},
    {"label": "VAL",     "residues": ["VAL"]},
]

# ----------------------------------------------------------------------
# Herramientas para PDB y deuteración
# ----------------------------------------------------------------------

def load_template(pdb_path: Path) -> gemmi.Structure:
    if not pdb_path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo: {pdb_path}")
    st = gemmi.read_structure(str(pdb_path))
    total_atoms = sum(len(residue) for model in st for chain in model for residue in chain)
    if total_atoms == 0:
        raise ValueError("El PDB no contiene átomos de proteína (ATOM).")
    log.info("Plantilla cargada: %d átomos proteicos.", total_atoms)
    return st

def compute_lability(residue: gemmi.Residue) -> List[bool]:
    labile = []
    prev_heavy = ""
    for atom in residue:
        elem = atom.element.name.upper()
        if elem in ("O", "N", "S"):
            labile.append(False)
            prev_heavy = elem
        elif elem in ("H", "D"):
            labile.append(prev_heavy in ("O", "N", "S"))
        else:
            prev_heavy = "C" if elem == "C" else ""
            labile.append(False)
    return labile

def build_deuterated_residue_set(pattern_bits: int, groups: List[dict]) -> set:
    deuterated = set()
    for i, group in enumerate(groups):
        if (pattern_bits >> i) & 1:
            for res in group["residues"]:
                deuterated.add(res)
    return deuterated

def apply_pattern(structure: gemmi.Structure, deuterated_residues: set) -> Tuple[gemmi.Structure, int]:
    """
    Devuelve la estructura con los H no lábiles sustituidos por D en los residuos indicados
    y el número de H no lábiles deuterados.
    """
    st = structure.clone()
    nl_count = 0
    for model in st:
        for chain in model:
            for residue in chain:
                resname = residue.name.strip().upper()
                if resname not in deuterated_residues:
                    continue
                labile = compute_lability(residue)
                for atom, is_labile in zip(residue, labile):
                    if atom.element.name == 'H' and not is_labile:
                        atom.name = atom.name.replace('H', 'D', 1)
                        atom.element = gemmi.Element('D')
                        nl_count += 1
    return st, nl_count

def apply_d2o_percentage(structure: gemmi.Structure, pct: int, rng: np.random.Generator) -> gemmi.Structure:
    if pct == 0:
        return structure
    labile_positions = []
    atom_idx = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                labile = compute_lability(residue)
                for atom, is_labile in zip(residue, labile):
                    if atom.element.name == 'H' and is_labile:
                        labile_positions.append((model, chain, residue, atom, atom_idx))
                    atom_idx += 1
    if not labile_positions:
        return structure
    k = int(len(labile_positions) * pct / 100)
    if k == 0:
        return structure
    if k == len(labile_positions):
        for _, _, _, atom, _ in labile_positions:
            atom.name = atom.name.replace('H', 'D', 1)
            atom.element = gemmi.Element('D')
        return structure
    rng.shuffle(labile_positions)
    for _, _, _, atom, _ in labile_positions[:k]:
        atom.name = atom.name.replace('H', 'D', 1)
        atom.element = gemmi.Element('D')
    return structure

def write_pdb_file(structure: gemmi.Structure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(path))

# ----------------------------------------------------------------------
# Comptage de deutérium
# ----------------------------------------------------------------------

def count_total_H(structure: gemmi.Structure) -> int:
    """
    Compte tous les atomes H de la structure fully protonée (template).
    Ce nombre sert de dénominateur pour le calcul du %D total.
    """
    count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name == 'H':
                        count += 1
    return count

def count_nonlabile_H(structure: gemmi.Structure) -> int:
    """
    Cuenta los átomos H NO lábiles en la estructura fully protonada.
    Sirve como denominador fijo para el %D no lábil.
    """
    count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                labile = compute_lability(residue)
                for atom, is_labile in zip(residue, labile):
                    if atom.element.name == 'H' and not is_labile:
                        count += 1
    return count

def count_D_atoms(structure: gemmi.Structure) -> int:
    """
    Compte les atomes D dans une structure après deutération
    (pattern + %D₂O).
    """
    count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name == 'D':
                        count += 1
    return count

def compute_deuterium_pct(n_D: int, n_H_total: int) -> float:
    if n_H_total == 0:
        return 0.0
    return float(n_D) / float(n_H_total) * 100.0

# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def pattern_string(bits: int, n_bits: int) -> str:
    return format(bits, f'0{n_bits}b')

def generate_percentages(step: int) -> List[int]:
    if step <= 0 or step > 100:
        raise ValueError("El step debe estar entre 1 y 100.")
    pcts = list(range(0, 101, step))
    if pcts[-1] != 100:
        pcts.append(100)
    return pcts

# ----------------------------------------------------------------------
# Pepsi-SANS
# ----------------------------------------------------------------------
def run_pepsi(job: Tuple[Path, float, Path, str]) -> Tuple[Path, int, str]:
    pdb_path, d2o_fraction, dat_path, pepsi_bin = job
    cmd = [
        pepsi_bin,
        str(pdb_path),
        "--hModel", "3",
        "--conc", "2.5",
        "--d2o", f"{d2o_fraction:.2f}",
        "-o", str(dat_path),
        "-ms", "0.32",
        # NOTE: -x (suppress log) intentionally removed so that Pepsi-SANS
        # writes a .log file alongside the .dat, from which we extract Rg.
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return dat_path, proc.returncode, proc.stderr.strip()
    except Exception as e:
        return dat_path, -1, str(e)

def run_all_jobs(jobs: List[Tuple], n_threads: int) -> None:
    if not jobs:
        return
    errors = []
    with ProcessPoolExecutor(max_workers=n_threads if n_threads > 0 else None) as ex:
        futures = {ex.submit(run_pepsi, job): job for job in jobs}
        for fut in as_completed(futures):
            dat_path, rc, stderr = fut.result()
            if rc != 0:
                errors.append(f"Pepsi-SANS falló ({rc}): {dat_path}\n{stderr}")
    if errors:
        raise RuntimeError(f"{len(errors)} trabajos de Pepsi-SANS fallaron:\n" + "\n".join(errors))

# ----------------------------------------------------------------------
# Parseo de .dat
# ----------------------------------------------------------------------
def parse_dat(path: Path, read_q: bool = False) -> Tuple[Optional[List[float]], List[float]]:
    q_vals, i_vals = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                i_vals.append(float(parts[1]))
                if read_q:
                    q_vals.append(float(parts[0]))
            except ValueError:
                continue
    if not read_q:
        q_vals = None
    return q_vals, i_vals

# ----------------------------------------------------------------------
# Parseo del .log de Pepsi-SANS para extraer Rg
# ----------------------------------------------------------------------
def parse_log_rg(log_path: Path) -> Optional[float]:
    """
    Extrae el Rg del contraste de longitud de dispersión del .log de Pepsi-SANS.

    Busca la línea:
        "Radius of gyration (the diff. of sc. length)........ : <valor> A"

    Si el valor es "-nan" (contraste nulo, p.ej. a la condición de match point),
    devuelve 0.0.  Devuelve None solo si el fichero no existe o la línea no
    aparece (warning no bloqueante).
    """
    if not log_path.exists():
        log.warning("Fichero .log no encontrado: %s", log_path)
        return None
    try:
        with open(log_path) as fh:
            for line in fh:
                m = _RG_PATTERN.match(line.strip())
                if m:
                    raw = m.group(1).lower()
                    if 'nan' in raw:
                        return 0.0
                    return float(raw)
    except Exception as exc:
        log.warning("Error al parsear %s: %s", log_path, exc)
    return None

# ----------------------------------------------------------------------
# Cálculo de fitness (multi-referencia, idéntico a fitness_evaluation.py)
# ----------------------------------------------------------------------

def compute_incoherent_background(d2o_percent: int) -> float:
    return -0.0117 * d2o_percent + 1.25

def compute_ratio(I: np.ndarray, d2o_pct: int) -> float:
    bg = compute_incoherent_background(d2o_pct)
    if bg <= 0:
        return 0.0
    return float(np.max(I) / bg)

def compute_product_areas(
    q_trunc: np.ndarray,
    I_trunc: np.ndarray,
    ref_curves: List[np.ndarray],
) -> float:
    product_areas = 1.0
    for I_ref in ref_curves:
        try:
            def linear_model(I, a, b):
                return a * I + b
            params, _ = curve_fit(linear_model, I_trunc, I_ref)
            a, b = params
            I_scaled = a * I_trunc + b
        except Exception:
            I_scaled = I_trunc
        area = simpson(np.abs(I_scaled - I_ref), x=q_trunc)
        product_areas *= area
    return float(product_areas)

def evaluate_curve_metrics(
    q_full: np.ndarray,
    I_full: np.ndarray,
    d2o_pct: int,
    ref_curves: List[np.ndarray],
    q_max: float,
    ratio_threshold: float,
) -> Tuple[float, float]:
    mask = q_full <= q_max
    q_trunc = q_full[mask]
    I_trunc = I_full[mask]

    ratio = compute_ratio(I_trunc, d2o_pct)
    if ratio < ratio_threshold:
        return 0.0, ratio

    product_areas = compute_product_areas(q_trunc, I_trunc, ref_curves)
    return product_areas, ratio

# ----------------------------------------------------------------------
# Escritura del HDF5
# ----------------------------------------------------------------------
class H5Writer:
    """
    Escribe incrementalmente el HDF5 de salida.

    Datasets (orden numérico):
      '0'  → patterns
      '1'  → d2o_pct
      '2'  → ratio
      '3'  → product_areas
      '4'  → Q
      '5'  → I
      '6'  → deuterium_pct      (%D total)
      '9'  → nonlabile_deut_pct (%D no lábil)
      '10' → rg                 (radio de giro en Å, del .log de Pepsi-SANS)
    """

    def __init__(self, path: Path, expected_curves: int):
        self.path = path
        self.expected_curves = expected_curves
        self.curves_written = 0
        self.q_written = False
        self.n_points: Optional[int] = None
        self._file = h5py.File(path, 'w')
        self._I_ds = None
        self._pat_ds = None
        self._d2o_ds = None
        self._product_areas_ds = None
        self._ratio_ds = None
        self._deut_ds = None
        self._nonlabile_ds = None
        self._rg_ds = None          # radio de giro por curva

    def write_q(self, q: list):
        if self.q_written:
            raise RuntimeError("El vector Q ya fue escrito.")
        self.n_points = len(q)
        self._file.create_dataset('Q', data=np.array(q, dtype='<f4'))
        self._I_ds = self._file.create_dataset(
            'I', shape=(self.expected_curves, self.n_points), dtype='<f4'
        )
        self._pat_ds = self._file.create_dataset(
            'patterns', shape=(self.expected_curves,), dtype=h5py.string_dtype()
        )
        self._d2o_ds = self._file.create_dataset(
            'd2o_pct', shape=(self.expected_curves,), dtype='int32'
        )
        self._deut_ds = self._file.create_dataset(
            'deuterium_pct', shape=(self.expected_curves,), dtype='<f4',
            fillvalue=np.nan,
        )
        self._nonlabile_ds = self._file.create_dataset(
            'nonlabile_deut_pct', shape=(self.expected_curves,), dtype='<f4',
            fillvalue=np.nan,
        )
        self._rg_ds = self._file.create_dataset(
            'rg', shape=(self.expected_curves,), dtype='<f4',
            fillvalue=np.nan,
        )
        self.q_written = True

    def create_product_areas_dataset(self):
        if self._product_areas_ds is not None:
            return
        if not self.q_written:
            raise RuntimeError("No se puede crear product_areas sin haber escrito Q.")
        self._product_areas_ds = self._file.create_dataset(
            'product_areas', shape=(self.expected_curves,), dtype='<f4', fillvalue=np.nan
        )
        self._ratio_ds = self._file.create_dataset(
            'ratio', shape=(self.expected_curves,), dtype='<f4', fillvalue=np.nan
        )

    def write_curve(self, I: list, pattern: str, d2o_pct: int,
                    deuterium_pct: float = np.nan,
                    nonlabile_deut_pct: float = np.nan,
                    rg: float = np.nan):
        if not self.q_written:
            raise RuntimeError("Debe escribirse Q antes de las curvas.")
        if self.curves_written >= self.expected_curves:
            raise RuntimeError("Se escribieron más curvas de las esperadas.")
        if len(I) != self.n_points:
            raise ValueError(f"Longitud {len(I)} != {self.n_points}")
        idx = self.curves_written
        self._I_ds[idx] = np.array(I, dtype='<f4')
        self._pat_ds[idx] = pattern
        self._d2o_ds[idx] = d2o_pct
        self._deut_ds[idx] = np.float32(deuterium_pct)
        self._nonlabile_ds[idx] = np.float32(nonlabile_deut_pct)
        self._rg_ds[idx] = np.float32(rg) if rg is not None else np.float32(np.nan)
        self.curves_written += 1

    def get_I(self, index: int) -> np.ndarray:
        return self._I_ds[index]

    def set_product_areas(self, index: int, product_areas: float, ratio: float):
        if self._product_areas_ds is None:
            raise RuntimeError("Los datasets de product_areas no están creados.")
        self._product_areas_ds[index] = np.float32(product_areas)
        self._ratio_ds[index] = ratio

    def finalize(self):
        if not self.q_written:
            raise RuntimeError("No se escribió el vector Q.")
        if self.curves_written != self.expected_curves:
            raise RuntimeError(
                f"Faltan curvas: esperadas {self.expected_curves}, escritas {self.curves_written}"
            )
        index_map = {
            '0':  'patterns',
            '1':  'd2o_pct',
            '2':  'ratio',
            '3':  'product_areas',
            '4':  'Q',
            '5':  'I',
            '6':  'deuterium_pct',
            '9':  'nonlabile_deut_pct',
            '10': 'rg',
        }
        for idx_str, name in index_map.items():
            if name in self._file:
                self._file[idx_str] = self._file[name]
        self._file.close()

# ----------------------------------------------------------------------
# Bucle principal
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fase 1: generación de curvas SANS.")
    parser.add_argument("--pdb", required=True, help="PDB de referencia")
    parser.add_argument("--mode", choices=["complete", "reduced"], default="reduced",
                        help="Modo de agrupación de AA (default: reduced)")
    parser.add_argument("--d2o-step", type=int, default=5,
                        help="Paso de porcentaje de D2O (1-100, default 5)")
    parser.add_argument("--output", default="output.h5", help="Archivo HDF5 de salida")
    parser.add_argument("--threads", type=int, default=0,
                        help="Número de hilos para Pepsi-SANS (0 = automático)")
    parser.add_argument("--pepsi", default="Pepsi-SANS", help="Ruta al ejecutable Pepsi-SANS")
    parser.add_argument("--work-dir", default=".", help="Directorio para carpetas temporales")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para reproducibilidad")
    parser.add_argument("--test-mode", action="store_true",
                        help="Procesa solo un patrón (reducido) y conserva los archivos temporales")
    parser.add_argument("--keep-files", action="store_true",
                        help="Conserva los directorios temporales tras cada patrón")
    parser.add_argument("--q-max-fitness", type=float, default=0.3, help="q máximo para el cálculo de fitness")
    parser.add_argument("--ratio-threshold", type=float, default=0.01, help="Umbral de ratio para fitness")
    args = parser.parse_args()

    if args.test_mode:
        args.mode = "reduced"
        if not args.keep_files:
            args.keep_files = True
        log.info("*** MODO DE PRUEBA: 1 patrón, archivos conservados ***")

    if not (1 <= args.d2o_step <= 100):
        sys.exit("--d2o-step debe estar entre 1 y 100.")
    percentages = generate_percentages(args.d2o_step)
    n_d2o = len(percentages)
    groups = COMPLETE_GROUPS if args.mode == "complete" else REDUCED_GROUPS
    n_bits = len(groups)
    n_patterns = 1 << n_bits
    n_curves_total = n_d2o if args.test_mode else n_patterns * n_d2o
    log.info("Modo: %s (%d bits, %d patrones)", args.mode, n_bits, n_patterns)
    log.info("Pasos D₂O: %d (step=%d) → %d curvas totales", n_d2o, args.d2o_step, n_curves_total)
    log.info("Salida: %s", args.output)

    # Cargar plantilla
    template = load_template(Path(args.pdb))

    # Conteos de H totales y no lábiles (denominadores fijos)
    total_H = count_total_H(template)
    total_nonlabile_H = count_nonlabile_H(template)
    log.info("Atomes H totales: %d  (denominador para %%D total)", total_H)
    log.info("Atomes H no lábiles: %d  (denominador para %%D no lábil)", total_nonlabile_H)
    if total_H == 0:
        log.warning("Ningún H en la plantilla. deuterium_pct será siempre 0.")
    if total_nonlabile_H == 0:
        log.warning("Ningún H no lábil en la plantilla. nonlabile_deut_pct será siempre 0.")

    # RNG para D2O
    rng = np.random.default_rng(args.seed if args.seed is not None else None)

    # Escritor HDF5
    writer = H5Writer(Path(args.output), n_curves_total)

    # Referencias internas (se construyen a partir del patrón 0)
    ref_curves: Optional[List[np.ndarray]] = None
    full_q: Optional[np.ndarray] = None

    # Procesar patrones
    first_pattern = True
    for pat_bits in range(n_patterns):
        if args.test_mode and pat_bits > 0:
            break
        pat_str = pattern_string(pat_bits, n_bits)
        t_start = time.time()
        log.info("Procesando patrón %s (%d/%d)", pat_str, pat_bits+1, n_patterns if not args.test_mode else 1)

        # Directorios temporales
        work = Path(args.work_dir)
        pdb_dir = work / f"pdbs_{pat_str}"
        simul_dir = work / f"simul_{pat_str}"
        pdb_dir.mkdir(parents=True, exist_ok=True)
        simul_dir.mkdir(parents=True, exist_ok=True)

        # Deuteración de patrón (no lábil) + contaje de H no lábiles deuterados
        deuterated_residues = build_deuterated_residue_set(pat_bits, groups)
        pattern_structure, nl_count = apply_pattern(template, deuterated_residues)

        # %D no lábil constante para todo el patrón
        if total_nonlabile_H > 0:
            nl_deut_pct = (nl_count / total_nonlabile_H) * 100.0
        else:
            nl_deut_pct = 0.0
        log.info("  %d H no lábiles deuterados → %.2f%% no lábil", nl_count, nl_deut_pct)

        # Generar PDBs + cálculo del %D total por condición
        jobs = []
        deut_pct_by_d2o: Dict[int, float] = {}

        for pct in percentages:
            pct_structure = pattern_structure.clone()
            pct_structure = apply_d2o_percentage(pct_structure, pct, rng)

            n_D = count_D_atoms(pct_structure)
            deut_pct_by_d2o[pct] = compute_deuterium_pct(n_D, total_H)

            pdb_path = pdb_dir / f"{pct}.pdb"
            write_pdb_file(pct_structure, pdb_path)
            dat_path = simul_dir / f"{pct}.dat"
            d2o_frac = pct / 100.0
            jobs.append((pdb_path, d2o_frac, dat_path, args.pepsi))

        # Ejecutar Pepsi-SANS (genera .dat y .log por cada condición)
        run_all_jobs(jobs, args.threads)

        # Parsear y escribir curvas
        for idx, pct in enumerate(percentages):
            dat_path = simul_dir / f"{pct}.dat"
            log_path = simul_dir / f"{pct}.log"   # generado por Pepsi-SANS (sin -x)

            if first_pattern and idx == 0:
                q_list, I_list = parse_dat(dat_path, read_q=True)
                writer.write_q(q_list)
                full_q = np.array(q_list)
            else:
                _, I_list = parse_dat(dat_path, read_q=False)

            # Extraer Rg del .log correspondiente
            rg_val = parse_log_rg(log_path)

            writer.write_curve(
                I_list, pat_str, pct,
                deuterium_pct=deut_pct_by_d2o[pct],
                nonlabile_deut_pct=nl_deut_pct,
                rg=rg_val,
            )
            curve_index = writer.curves_written - 1

            # Si ya tenemos referencias, calcular ratio y product_areas inmediatamente
            if ref_curves is not None and full_q is not None:
                product_areas, ratio_val = evaluate_curve_metrics(
                    full_q, np.array(I_list), int(pct),
                    ref_curves, args.q_max_fitness, args.ratio_threshold,
                )
                if writer._product_areas_ds is None:
                    writer.create_product_areas_dataset()
                writer.set_product_areas(curve_index, product_areas, ratio_val)

        # Patrón 0: capturar referencias y calcular ratio/product_areas pour toutes ses courbes
        if pat_bits == 0:
            start_idx = writer.curves_written - n_d2o
            idx_0 = start_idx + percentages.index(0)
            idx_100 = start_idx + percentages.index(100)

            mask = full_q <= args.q_max_fitness
            I0 = writer.get_I(idx_0)[mask]
            I100 = writer.get_I(idx_100)[mask]
            ref_curves = [I0, I100]
            log.info("Referencias internas capturadas: patrón 0, 0%% y 100%% D₂O")

            writer.create_product_areas_dataset()
            for c_idx in range(start_idx, writer.curves_written):
                I_full = writer.get_I(c_idx)
                pct_val = writer._d2o_ds[c_idx]
                product_areas, ratio_val = evaluate_curve_metrics(
                    full_q, I_full, int(pct_val),
                    ref_curves, args.q_max_fitness, args.ratio_threshold,
                )
                writer.set_product_areas(c_idx, product_areas, ratio_val)
            log.info("Ratio y product_areas calculados pour toutes les courbes du pattern 0.")

        first_pattern = False

        # Limpieza — shutil.rmtree elimina también los .log junto a los .dat
        if not args.keep_files:
            shutil.rmtree(pdb_dir, ignore_errors=True)
            shutil.rmtree(simul_dir, ignore_errors=True)
        else:
            log.info("Conservados: %s y %s", pdb_dir, simul_dir)

        elapsed = time.time() - t_start
        log.info("Patrón %s completado en %.1f s", pat_str, elapsed)

    writer.finalize()
    log.info("Pipeline completado. HDF5 escrito en %s", args.output)

if __name__ == "__main__":
    main()
