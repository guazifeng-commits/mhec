#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aimd_postproc.py
======================

Comprehensive post-processing tool for VASP AIMD results focused on
viscosity estimation (Green-Kubo) and Stokes-Einstein (via diffusion),
and further calculation of friction coefficient. Also computes RDF, MSD, VACF,
energy/temperature time series, and many intermediate diagnostics. Saves results
in Origin-friendly .dat/.csv files and generates PNG plots.

Features (comprehensive analysis suite):

Basic Analysis:
 - Read trajectories from XDATCAR (preferred) or vasprun.xml frames
 - Read stress & energy information from OUTCAR or vasprun.xml
 - Compute total RDF g(r) with PBC handling and output first-peak and CN
 - Compute MSD, unwrapping trajectories, estimate D by linear fit
 - Compute VACF (from finite-difference velocities) and vibrational info
 - Stokes-Einstein viscosity using user-specified R (or auto from RDF)
 - Green-Kubo viscosity via stress autocorrelation (FFT-based ACF),
   block-averaging, bootstrap CI, windowing and truncation strategies
 - Calculate TWO types of friction coefficients:
   * Macroscopic friction coefficient mu = eta*U/(p*h) for lubrication analysis
   * Molecular friction coefficient xi = kBT/D for atomic mobility analysis

Advanced Structural Analysis (--advanced_analysis or individual flags):
 - Partial RDFs g_ij(r) for all species pairs (--partial_rdf)
 - Static structure factor S(q) for comparison with experiments (--structure_factor)
 - Partial coordination numbers for chemical environment analysis (--coordination_analysis)
 - Bond length and angle distributions (--bond_analysis)

Advanced Dynamical Analysis:
 - MSD higher-order moments for dynamic heterogeneity (--msd_moments)
 - Non-Gaussian parameter alpha2 = <r^4>/(3<r^2>^2) - 1 for diffusion uniformity
 - Individual species MSD and diffusion coefficients

Output & Visualization:
 - Extensive outputs: .dat/.csv files (Origin-compatible), PNG plots, JSON summary
 - Console report with all key results and statistical uncertainties

Notes & assumptions
 - The script assumes XDATCAR contains direct (fractional) coordinates
   and lattice vectors in standard VASP format. If you only have
   vasprun.xml or OUTCAR, the script can extract stress and (some) energies
   but XDATCAR is strongly recommended for RDF/MSD analysis.
 - The script will attempt to infer element types and masses from
   POSCAR/XDATCAR header. If not available, user can supply --masses.

Usage examples:

1. Basic analysis (RDF, MSD, viscosity, friction):
  python aimd_postproc.py --xdatcar XDATCAR --outcar OUTCAR \
      --T 300 --potim 1.0 --r_eff 3.75 --n_blocks 5 --stable_frac 0.3 \
      --out_prefix basic --bootstrap --tmax_ps 20

2. Complete analysis with all advanced features:
  python aimd_postproc.py --xdatcar XDATCAR --outcar OUTCAR \
      --T 300 --potim 1.0 --r_eff 3.75 --n_blocks 5 --stable_frac 0.3 \
      --out_prefix complete --bootstrap --tmax_ps 20 --window hann \
      --friction_U "0.01 0.1 1.0" --friction_p 1e7 --friction_h 5e-9 \
      --advanced_analysis --qmax 20.0 --bond_cutoff 3.5

3. Selective advanced analysis:
  python aimd_postproc.py --xdatcar XDATCAR --outcar OUTCAR \
      --T 300 --potim 1.0 --out_prefix selective \
      --partial_rdf --structure_factor --msd_moments \
      --coordination_analysis --bond_analysis

Advanced analysis features:
  --advanced_analysis     : Enable all advanced structural and dynamical analysis
  --partial_rdf          : Compute partial RDFs g_ij(r) for all species pairs
  --structure_factor     : Compute static structure factor S(q)
  --coordination_analysis: Compute partial coordination numbers
  --msd_moments         : Compute MSD higher moments and non-Gaussian parameter alpha2
  --bond_analysis       : Compute bond length and angle distributions
  --qmax 20.0           : Maximum q value for structure factor (A^-1)
  --bond_cutoff 3.5     : Cutoff distance for bond analysis (A)
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd # Added for friction calculation
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
from scipy.stats import linregress

# Lazy import matplotlib to allow running in headless mode if not needed
try:
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境也能出图
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# 进度条 (作为包导入时可用；独立脚本运行时退化为无进度)
try:
    from ..progress import track, ProgressBar
except Exception:
    def track(iterable, desc="", total=None, stream=None):
        for x in iterable:
            yield x
    ProgressBar = None

# 双语支持 (作为包导入时可用; 独立脚本运行时退化为中文)
try:
    from ..i18n import T
except Exception:
    def T(zh, en=None):
        return zh

# -----------------------------
# Physical constants & tables
# -----------------------------
kB = 1.380649e-23  # J/K
fs2s = 1e-15
A32m3 = 1e-30
kbar2Pa = 1e8

# Simple atomic mass table (g/mol). 统一使用 mhec.periodic 的元素表 (元素更全、便于维护)。
# 独立脚本方式运行时退化到本地最小表。
try:
    from ..periodic import ELEMENTS as _ELEMENTS
    ATOMIC_MASSES = {sym: mass for sym, (_z, mass) in _ELEMENTS.items()}
except Exception:
    ATOMIC_MASSES = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'Al': 26.982,
        'Si': 28.085, 'Ga': 69.723, 'Ge': 72.630, 'In': 114.818, 'Sn': 118.710,
    }

# -----------------------------
# Helper I/O and parsers
# -----------------------------

def read_poscar_header(poscar_path: str) -> Tuple[List[str], List[int], float, np.ndarray]:
    """Parse header lines of POSCAR/XDATCAR to extract element names, counts,
    scale and lattice vectors. Returns (elements, counts, scale, lattice_array).
    If element names are missing, returns empty list for elements.
    """
    with open(poscar_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.rstrip() for l in f.readlines()]

    # Some POSCAR variants: line 1 comment, line2 scale, line3-5 lattice
    # line6 may be element symbols or counts. Determine heuristically.
    if len(lines) < 5:
        raise RuntimeError(f"POSCAR/XDATCAR {poscar_path} too short")

    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale

    # Check line 5 and 6
    line5 = lines[5].split()
    elems = []
    counts = []
    # If tokens are alphabetic, assume they are element symbols
    if all(re.match(r"^[A-Za-z]{1,2}$", tok) for tok in line5):
        elems = line5
        counts = [int(x) for x in lines[6].split()]
    else:
        # line5 is counts, but we might find element symbols at line 0 comment
        try:
            counts = [int(x) for x in line5]
            # try to extract element symbols from line0 comment (rare)
            # default to empty
            elems = []
        except Exception:
            counts = []
            elems = []

    return elems, counts, scale, lattice


def read_xdatcar_frames(xdatcar_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """Parse XDATCAR and return fractional coordinates array of shape (nframes, natoms, 3)
    and lattice (3x3). Also returns elements and counts when present.
    This parser is forgiving: it scans for sequences of natom coordinate lines.
    """
    with open(xdatcar_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.rstrip() for l in f.readlines()]

    if len(lines) < 8:
        raise RuntimeError('XDATCAR too short')

    elems = []
    counts = []
    try:
        scale = float(lines[1].split()[0])
        lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale
    except Exception as e:
        raise RuntimeError('Cannot parse lattice from XDATCAR: ' + str(e))

    # Try to read element symbols & counts if present in first 8 lines
    tok5 = lines[5].split()
    if all(re.match(r"^[A-Za-z]{1,2}$", tok) for tok in tok5):
        elems = tok5
        counts = [int(x) for x in lines[6].split()]
        coord_start = 7
    else:
        # If not present, the counts are on line 5
        try:
            counts = [int(x) for x in tok5]
            # elems unknown
            elems = []
            coord_start = 6
        except Exception:
            # fallback: find first occurrence of a coordinate-like block
            coord_start = 6

    natoms = sum(counts) if len(counts) > 0 else None

    # Now find frames: lines containing 3 floats repeated natoms times
    coords_list = []
    i = coord_start
    # collect all numeric coordinate lines
    numeric_lines = []
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 3 and all(re.match(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$", p) for p in parts[:3]):
            numeric_lines.append(lines[i])
        i += 1

    if natoms is None:
        # try to infer natoms by looking for repetition blocks; assume a consistent block size
        # naive approach: if total numeric lines divisible by some small integer, choose that
        NL = len(numeric_lines)
        # here we can't automatically know; we assume user provided natoms via counts earlier
        raise RuntimeError('Could not infer natoms from XDATCAR. Please provide POSCAR with species/counts or supply natoms.')

    nframes = len(numeric_lines) // natoms
    if nframes * natoms != len(numeric_lines):
        # ignore trailing partial frame
        nframes = int(nframes)

    coords = np.zeros((nframes, natoms, 3), dtype=float)
    for f in range(nframes):
        for a in range(natoms):
            line = numeric_lines[f * natoms + a].split()
            coords[f, a, :] = [float(line[0]), float(line[1]), float(line[2])]

    return coords, lattice, elems, counts


def read_vasprun_stress_energy(vasprun_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], List[float]]:
    """Parse vasprun.xml for stress and optional volume, energies per ionic step.
    Returns stress_array (nsteps,6), volume_list (or None), energies(list) in eV.
    
    Supports both standard DFT and MLFF run output formats.
    For large files, uses iterparse to avoid memory issues.
    """
    import xml.etree.ElementTree as ET

    stress_list = []
    vol_list = []
    energies = []

    try:
        # Use iterparse for large files (MLFF run can produce huge vasprun.xml)
        in_calc = False
        current_stress = None
        current_vol = None
        current_energy = None
        in_stress_varray = False
        stress_rows = []

        for event, elem in ET.iterparse(vasprun_path, events=('start', 'end')):
            tag = elem.tag

            if event == 'start':
                if tag == 'calculation':
                    in_calc = True
                    current_stress = None
                    current_vol = None
                    current_energy = None
                    stress_rows = []
                    in_stress_varray = False
                elif tag == 'varray' and elem.get('name') == 'stress':
                    in_stress_varray = True
                    stress_rows = []

            elif event == 'end':
                if tag == 'v' and in_stress_varray:
                    if elem.text:
                        try:
                            stress_rows.append([float(x) for x in elem.text.strip().split()])
                        except Exception:
                            pass

                elif tag == 'varray' and in_stress_varray:
                    in_stress_varray = False
                    if len(stress_rows) >= 3:
                        s = np.array(stress_rows[:3])
                        current_stress = np.array([
                            s[0, 0], s[1, 1], s[2, 2],
                            s[0, 1], s[1, 2], s[2, 0]
                        ]) * kbar2Pa

                elif tag == 'i' and in_calc:
                    name = elem.get('name', '')
                    if name == 'e_fr_energy' and elem.text:
                        try:
                            current_energy = float(elem.text.strip())
                        except Exception:
                            pass
                    elif name == 'volume' and elem.text:
                        try:
                            current_vol = float(elem.text.strip())
                        except Exception:
                            pass

                elif tag == 'calculation':
                    in_calc = False
                    if current_stress is not None:
                        stress_list.append(current_stress)
                    if current_vol is not None:
                        vol_list.append(current_vol)
                    if current_energy is not None:
                        energies.append(current_energy)
                    # Free memory for large files
                    elem.clear()

    except ET.ParseError as e:
        print(f'Warning: vasprun.xml parse error (possibly incomplete file): {e}')
    except Exception as e:
        print(f'Warning: Error reading vasprun.xml: {e}')

    stress_arr = np.array(stress_list) if len(stress_list) > 0 else np.empty((0, 6))
    vol_arr = np.array(vol_list) if len(vol_list) > 0 else None
    return stress_arr, vol_arr, energies


def read_outcar_stress_energy(outcar_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], List[float], List[float]]:
    """Parse OUTCAR for stresses, volume, energies and temperatures per ionic step.
    Returns stress_array (nsteps,6) in Pa, volume_list (A^3) or None, energies (eV), temps (K)

    支持两种应力格式:
    1. 标准 DFT: 'in kB' 行 (单位 kBar)
    2. MLFF run: 'ML FORCE on cell =-STRESS' 块中的 'Total:' 行 (单位 eV/cell)
       转换: σ(Pa) = -Total(eV) / V(m³), 其中 1 eV = 1.602176634e-19 J

    用 'LOOP+' 作为离子步结束标记。
    """
    eV_to_J = 1.602176634e-19

    stresses = []
    volumes = []
    energies = []
    temps = []

    current_volume = None
    current_energy = None
    current_stress = None
    current_temp = None
    in_ml_stress_block = False  # 标记是否在 ML STRESS 块内

    def _commit_step():
        nonlocal current_energy, current_stress, current_temp
        if current_energy is not None:
            energies.append(current_energy)
        if current_stress is not None:
            stresses.append(current_stress)
        if current_temp is not None:
            temps.append(current_temp)
        if current_volume is not None and current_stress is not None:
            volumes.append(current_volume)
        current_energy = None
        current_stress = None
        current_temp = None

    with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 能量: TOTEN (DFT 模式)
            if 'free  energy   TOTEN' in line:
                try:
                    current_energy = float(line.split()[-2])
                except Exception:
                    pass

            # 体积
            if 'volume of cell' in line.lower():
                try:
                    current_volume = float(line.split()[-1])
                except Exception:
                    pass

            # 标准 DFT 应力: 'in kB' 行
            if 'in kB' in line:
                parts = line.split()
                floats = []
                for tok in parts:
                    try:
                        floats.append(float(tok))
                    except Exception:
                        pass
                if len(floats) >= 6:
                    xx, yy, zz, xy, yz, zx = floats[-6:]
                    current_stress = [xx * kbar2Pa, yy * kbar2Pa, zz * kbar2Pa,
                                      xy * kbar2Pa, yz * kbar2Pa, zx * kbar2Pa]

            # MLFF 应力: 'ML FORCE on cell =-STRESS' 块
            if 'FORCE on cell =-STRESS' in line:
                in_ml_stress_block = True

            if in_ml_stress_block and line.strip().startswith('Total:'):
                in_ml_stress_block = False
                parts = line.split()
                # Total:  xx  yy  zz  xy  yz  zx
                if len(parts) >= 7:
                    try:
                        vals = [float(parts[i]) for i in range(1, 7)]
                        # FORCE on cell = -STRESS, 所以 stress = -force/V
                        # vals 单位是 eV, stress(Pa) = -vals(eV) * eV_to_J / V(m³)
                        if current_volume is not None and current_volume > 0:
                            V_m3 = current_volume * A32m3
                            current_stress = [
                                -vals[0] * eV_to_J / V_m3,
                                -vals[1] * eV_to_J / V_m3,
                                -vals[2] * eV_to_J / V_m3,
                                -vals[3] * eV_to_J / V_m3,
                                -vals[4] * eV_to_J / V_m3,
                                -vals[5] * eV_to_J / V_m3,
                            ]
                    except Exception:
                        pass

            # 温度
            if 'kin. lattice' in line.lower():
                m = re.search(r'temperature\s+([0-9]+\.?[0-9]*)\s*K', line, re.IGNORECASE)
                if m:
                    try:
                        current_temp = float(m.group(1))
                    except Exception:
                        pass

            # 离子步结束标记
            if 'LOOP+' in line:
                if current_energy is not None or current_stress is not None:
                    _commit_step()

    # 提交最后一步
    if current_energy is not None or current_stress is not None:
        _commit_step()

    stress_arr = np.array(stresses) if len(stresses) > 0 else np.empty((0, 6))
    vol_arr = np.array(volumes) if len(volumes) > 0 else None
    return stress_arr, vol_arr, energies, temps


def read_oszicar_energy(oszicar_path: str, potim_fs: float = 1.0):
    """
    针对 AIMD 格式的 OSZICAR:
        step   T= ...   E= ...   F= ...   E0= ...
    提取 step, 温度 T, 自由能 F。
    返回: energies (list of float), times_s (numpy array), temps (list of float)
    energies -> F= 列 (eV)
    times_s  -> 根据 step * POTIM 推算 (秒)
    temps    -> T= 列 (K)
    """
    steps, energies, temps = [], [], []
    with open(oszicar_path, "r", encoding='utf-8', errors="ignore") as f:
        for line in f:
            if " F=" in line and " T=" in line:
                try:
                    parts = line.split()
                    step = int(parts[0])
                    # 用关键字定位而非固定列号
                    T_val = None
                    F_val = None
                    for j, tok in enumerate(parts):
                        if tok == "T=" and j + 1 < len(parts):
                            T_val = float(parts[j + 1])
                        if tok == "F=" and j + 1 < len(parts):
                            F_val = float(parts[j + 1])
                    if T_val is not None and F_val is not None:
                        steps.append(step)
                        temps.append(T_val)
                        energies.append(F_val)
                except Exception:
                    continue

    if len(steps) == 0:
        return [], np.array([]), []

    times_fs = [(s - steps[0]) * potim_fs for s in steps]
    times_s = np.array(times_fs) * fs2s
    return energies, times_s, temps


# -----------------------------
# Analysis utilities
# -----------------------------

def fractional_to_cart(frac_coords: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert fractional coords (n, natoms, 3) to cartesian using lattice (3x3)
    If input is 2D (natoms,3), it still works.
    """
    arr = np.asarray(frac_coords)
    shape = arr.shape
    flat = arr.reshape(-1, 3)
    cart = flat.dot(lattice)
    return cart.reshape(shape)


def unwrap_positions(frac_coords: np.ndarray) -> np.ndarray:
    """Perform unwrapping in fractional coordinates to create continuous trajectories.
    frac_coords shape: (nframes, natoms, 3). Returns unwrapped fractional coords same shape.
    """
    nframes, natoms, _ = frac_coords.shape
    unwrapped = np.zeros_like(frac_coords)
    unwrapped[0] = frac_coords[0]
    for t in range(1, nframes):
        delta = frac_coords[t] - frac_coords[t - 1]
        # bring to [-0.5, 0.5) by rounding
        delta_wrapped = delta - np.rint(delta)
        unwrapped[t] = unwrapped[t - 1] + delta_wrapped
    return unwrapped


def compute_rdf(frac_coords: np.ndarray, lattice: np.ndarray, rmax: float = 10.0, nbins: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial distribution function g(r). frac_coords in fractional coords.
    lattice in A (3x3). rmax in A.
    Returns r (bin centers) and g(r).
    Optimization: Use numpy broadcasting and vectorization for pairwise distances.
    """
    nframes, natoms, _ = frac_coords.shape

    Vcell = abs(np.linalg.det(lattice))
    rho = natoms / Vcell

    dr = rmax / nbins
    edges = np.linspace(0.0, rmax, nbins + 1)
    hist = np.zeros(nbins)

    for f in track(range(nframes), desc="RDF 逐帧统计", total=nframes):
        pos = frac_coords[f] # Fractional coordinates for current frame
        
        # Compute all pairwise differences in fractional coordinates
        # (natoms, 1, 3) - (1, natoms, 3) -> (natoms, natoms, 3)
        diffs_frac = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        
        # Apply Minimum Image Convention (MIC) in fractional coordinates
        diffs_frac_mic = diffs_frac - np.rint(diffs_frac)
        
        # Convert MIC fractional differences to Cartesian differences
        diffs_cart_mic = np.dot(diffs_frac_mic, lattice)
        
        # Calculate Euclidean distances
        distances = np.linalg.norm(diffs_cart_mic, axis=-1)

        # Only consider upper triangle to avoid double counting and self-interaction
        # k=1 excludes self-interaction (diagonal elements)
        distances_upper_tri = distances[np.triu_indices(natoms, k=1)]

        # Populate histogram
        hist_frame, _ = np.histogram(distances_upper_tri, bins=edges)
        hist += hist_frame

    # Correct normalization for RDF
    r = (edges[:-1] + edges[1:]) / 2.0
    shell_vol = 4.0 * np.pi * r**2 * dr
    
    # Correct RDF normalization:
    # g(r) = (observed pairs in shell) / (expected pairs for ideal gas)
    # 
    # For each frame:
    # - We count pairs in upper triangle: natoms*(natoms-1)/2 total pairs
    # - Expected pairs in shell dr at distance r for ideal gas:
    #   = (total_pairs) * (shell_volume / total_volume)
    #   = natoms*(natoms-1)/2 * (4*pi*r^2*dr) / V
    # 
    # So normalization factor = nframes * natoms*(natoms-1)/2 * shell_vol / V
    
    total_pairs_per_frame = natoms * (natoms - 1) / 2.0
    
    g_r = np.zeros_like(hist, dtype=float)
    valid_indices = shell_vol > 1e-12 # Small threshold to avoid numerical issues
    
    # Expected number of pairs in each shell for ideal gas
    expected_pairs = nframes * total_pairs_per_frame * shell_vol[valid_indices] / Vcell
    
    g_r[valid_indices] = hist[valid_indices] / expected_pairs

    return r, g_r


def find_rdf_first_peak(r: np.ndarray, g_r: np.ndarray) -> Tuple[float, float]:
    """Return position and height of first peak in g(r). Simple local max search.
    """
    # skip first very small r near zero
    start = max(1, int(0.5 / (r[1] - r[0])))
    peak_idx = np.argmax(g_r[start:]) + start
    return float(r[peak_idx]), float(g_r[peak_idx])


def compute_coordination_number(r: np.ndarray, g_r: np.ndarray, rho: float, r_cutoff: float) -> float:
    """Compute coordination number by integrating RDF up to r_cutoff.
    CN = 4*pi * rho * integral[0 to r_cutoff] g(r) * r^2 dr
    """
    # Find index corresponding to r_cutoff
    cutoff_idx = np.searchsorted(r, r_cutoff)
    if cutoff_idx >= len(r):
        cutoff_idx = len(r) - 1
    
    # Integrate using trapezoidal rule
    r_int = r[:cutoff_idx+1]
    g_int = g_r[:cutoff_idx+1]
    integrand = g_int * r_int**2
    
    # Numerical integration
    dr = r_int[1] - r_int[0] if len(r_int) > 1 else 0
    integral = np.trapz(integrand, dx=dr)
    
    coordination_number = 4.0 * np.pi * rho * integral
    return float(coordination_number)


def find_rdf_first_minimum(r: np.ndarray, g_r: np.ndarray, peak_r: float) -> float:
    """Find the first minimum after the first peak for coordination number calculation.
    """
    # Find peak index
    peak_idx = np.argmin(np.abs(r - peak_r))
    
    # Look for minimum after the peak
    for i in range(peak_idx + 1, len(g_r) - 1):
        if g_r[i-1] > g_r[i] < g_r[i+1]:  # Local minimum
            return float(r[i])
    
    # If no minimum found, use 1.5 * peak_r as default
    return float(1.5 * peak_r)


def compute_msd(frac_coords: np.ndarray, lattice: np.ndarray, dt_fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MSD (mean squared displacement) using unwrapped positions.
    Returns times (s) and msd (Å²) arrays.
    Note: caller must convert Å² → m² if needed (multiply by 1e-20).
    """
    nframes, natoms, _ = frac_coords.shape
    unwrapped = unwrap_positions(frac_coords)
    cart = fractional_to_cart(unwrapped, lattice)

    msd = np.zeros(nframes)
    # All-to-all method for MSD
    for tau in track(range(1, nframes), desc="MSD 时间平均", total=nframes - 1):
        displacements = cart[tau:] - cart[:-tau]
        msd[tau] = np.mean(np.sum(displacements**2, axis=2))

    times = np.arange(nframes) * dt_fs * fs2s
    return times, msd


def compute_msd_by_species(frac_coords: np.ndarray, lattice: np.ndarray, species: List[str], dt_fs: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute MSD separately for each species (element).
    Returns times (s) and a dict mapping species -> msd_array (units: A^2, same as compute_msd raw).
    species: list of length natoms giving element labels (e.g., ['Li','Li','O',...])
    """
    nframes, natoms, _ = frac_coords.shape
    if len(species) != natoms:
        raise RuntimeError("Species length does not match number of atoms (natoms). Cannot compute per-species MSD.")
    unwrapped = unwrap_positions(frac_coords)
    cart = fractional_to_cart(unwrapped, lattice)  # in A

    # Group indices by species
    species_indices = defaultdict(list)
    for ai, s in enumerate(species):
        species_indices[s].append(ai)

    times = np.arange(nframes) * dt_fs * fs2s
    msd_species = {}
    for sname, indices in species_indices.items():
        idx_arr = np.array(indices, dtype=int)
        # Compute MSD for this species
        msd_s = np.zeros(nframes)
        # If there's only one atom of the species, MSD reduces to mean of squared displacement of that atom (works)
        for tau in range(1, nframes):
            # shape: (nsteps, n_selected_atoms, 3)
            displacements = cart[tau:, idx_arr, :] - cart[:-tau, idx_arr, :]
            msd_s[tau] = np.mean(np.sum(displacements**2, axis=2))
        msd_species[sname] = msd_s

    return times, msd_species


def estimate_diffusion_from_msd(times: np.ndarray, msd: np.ndarray, fit_start_frac: float = 0.2, fit_end_frac: float = 0.8) -> Tuple[float, float, float]:
    """Estimate diffusion coefficient D by linear fit to MSD(t) in a window.
    Returns (D, slope, intercept) where slope = 6D (for 3D diffusion).
    """
    n = len(times)
    i0 = int(n * fit_start_frac)
    i1 = int(n * fit_end_frac)
    if i1 <= i0 + 3:
        i0 = max(1, n // 4)
        i1 = n - 1
    x = times[i0:i1]
    y = msd[i0:i1]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    D = slope / 6.0
    return float(D), float(slope), float(intercept)


def compute_vacf_from_positions(cart_coords: np.ndarray, dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute VACF by finite-difference velocities from cartesian coords.
    cart_coords shape: (nframes, natoms, 3) in Angstrom
    dt_s: time step in seconds
    Returns lags (s) and vacf (m^2/s^2)
    """
    nframes, natoms, _ = cart_coords.shape

    # Convert positions from Å to m for correct velocity units
    cart_m = cart_coords * 1e-10  # Å → m

    # Calculate velocities using central difference (m/s)
    velocities = np.zeros_like(cart_m)
    velocities[1:-1] = (cart_m[2:] - cart_m[:-2]) / (2 * dt_s)
    velocities[0] = (cart_m[1] - cart_m[0]) / dt_s
    velocities[-1] = (cart_m[-1] - cart_m[-2]) / dt_s

    # Flatten velocities for FFT-based autocorrelation
    velocities_flat = velocities.reshape(nframes, -1)  # (nframes, natoms*3)

    n_usable = max(1, int(nframes * 0.9))  # Use 90% of lags
    vacf_sum = np.zeros(n_usable)

    nfft = 2**math.ceil(math.log2(2*nframes - 1))

    for i in range(velocities_flat.shape[1]):
        v_component = velocities_flat[:, i]

        padded = np.zeros(nfft)
        padded[:nframes] = v_component

        fft_v = np.fft.rfft(padded)
        power = fft_v * np.conj(fft_v)
        acf_full = np.fft.irfft(power, n=nfft)

        # Unbiased normalization, only first half
        for k in range(n_usable):
            vacf_sum[k] += acf_full[k] / (nframes - k)

    # Average over all components (natoms * 3)
    vacf = vacf_sum / (natoms * 3)

    lags = np.arange(n_usable) * dt_s
    return lags, vacf


def compute_pdos_from_vacf(vacf_times: np.ndarray, vacf: np.ndarray, 
                          window: str = 'hann', max_freq_THz: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 VACF 计算声子态密度（Phonon Density of States, PDOS）
    
    理论基础：
    根据 Wiener-Khinchin 定理，功率谱密度是自相关函数的傅里叶变换：
    PDOS(ω) = FT[VACF(t)]
    
    Parameters:
    -----------
    vacf_times : np.ndarray
        VACF 的时间数组（单位：秒）
    vacf : np.ndarray
        VACF 值（单位：m^2/s^2）
    window : str
        窗函数类型，用于减少频谱泄漏
        选项：'hann', 'hamming', 'blackman', 'none'
    max_freq_THz : float
        最大显示频率（THz）
    
    Returns:
    --------
    frequencies_THz : np.ndarray
        频率数组（THz）
    pdos : np.ndarray
        归一化的声子态密度
    """
    # 获取时间步长
    dt = vacf_times[1] - vacf_times[0] if len(vacf_times) > 1 else 1e-15
    
    # 应用窗函数以减少频谱泄漏
    if window == 'hann':
        window_func = np.hanning(len(vacf))
    elif window == 'hamming':
        window_func = np.hamming(len(vacf))
    elif window == 'blackman':
        window_func = np.blackman(len(vacf))
    else:
        window_func = np.ones(len(vacf))
    
    vacf_windowed = vacf * window_func
    
    # PDOS = Re[FT[VACF(t)]]
    # 根据 Wiener-Khinchin 定理，功率谱密度是自相关函数的傅里叶变换的实部
    # 注意：不是 |FT|²，那是对原始信号的功率谱
    fft_result = np.fft.rfft(vacf_windowed)
    
    # 取实部（余弦变换），乘以 dt 得到正确的谱密度量纲
    pdos_raw = np.abs(fft_result.real) * dt
    
    # 频率数组（Hz）
    frequencies_Hz = np.fft.rfftfreq(len(vacf), d=dt)
    
    # 转换为 THz
    frequencies_THz = frequencies_Hz * 1e-12
    
    # 只保留到 max_freq_THz
    mask = frequencies_THz <= max_freq_THz
    frequencies_THz = frequencies_THz[mask]
    pdos_raw = pdos_raw[mask]
    
    # 归一化 PDOS（使积分为 1）
    if len(pdos_raw) > 1:
        integral = np.trapz(pdos_raw, frequencies_THz)
        if integral > 0:
            pdos = pdos_raw / integral
        else:
            pdos = pdos_raw
    else:
        pdos = pdos_raw
    
    return frequencies_THz, pdos


def autocorr_fft(x: np.ndarray) -> np.ndarray:
    """Compute autocorrelation function of a 1D signal using FFT.
    
    Returns the full (non-centered) autocorrelation: C(τ) = <x(t) * x(t+τ)>
    This is the correct form for Green-Kubo integration.
    
    For Green-Kubo viscosity, we need the stress autocorrelation (NOT covariance),
    because the formula is η = (V/kBT) ∫ <σ(0)σ(t)> dt.
    The mean of off-diagonal stress should be ~0 at equilibrium, but we do NOT
    subtract it to avoid introducing artifacts in finite trajectories.
    """
    N = len(x)
    
    # Do NOT subtract mean — Green-Kubo requires full autocorrelation
    # For equilibrium off-diagonal stress, <σ_xy> ≈ 0 anyway
    
    # Pad with zeros to avoid circular convolution
    nfft = 2**math.ceil(math.log2(2*N - 1))
    padded = np.zeros(nfft)
    padded[:N] = x
    
    # FFT-based autocorrelation
    fft_x = np.fft.rfft(padded)
    power = fft_x * np.conj(fft_x)
    acf_full = np.fft.irfft(power, n=nfft)
    
    # Normalize by number of overlapping pairs (unbiased estimator)
    # Use first N-1 lags but apply a reliability cutoff:
    # at lag k, there are (N-k) overlapping pairs. We keep lags where
    # at least 10% of the data contributes, i.e. k < 0.9*N
    n_usable = max(1, int(N * 0.9))
    acf_result = np.empty(n_usable)
    for k in range(n_usable):
        acf_result[k] = acf_full[k] / (N - k)
    
    return acf_result


def compute_gk_viscosity_from_stress(stress_block: np.ndarray, V_m3: float, T: float, dt_s: float,
                                     unbiased_acf: bool = True, window: Optional[str] = None,
                                     truncation: Optional[str] = None, tmax_ps: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Compute Green-Kubo viscosity from a block of stress tensor components.
    stress_block: (nsteps, 6) array of [Pxx, Pyy, Pzz, Pxy, Pyz, Pzx] in Pa.
    V_m3: Volume in m^3.
    T: Temperature in K.
    dt_s: Time step in seconds.
    Returns (times, running_viscosity, truncation_index, sacf).
    sacf: stress autocorrelation function (averaged over off-diagonal components).

    Green-Kubo formula:
        η = (V / kBT) ∫₀^∞ <σ_αβ(0) σ_αβ(t)> dt

    For isotropic systems, average over the 3 independent off-diagonal components
    (xy, yz, zx) and optionally include the 3 normal-stress differences:
        σ'_xx = (Pxx - Pyy) / 2,  σ'_yy = (Pyy - Pzz) / 2,  σ'_zz = (Pzz - Pxx) / 2

    Using all 5 independent components improves statistics.
    """
    nsteps = stress_block.shape[0]

    # Off-diagonal components
    Pxy = stress_block[:, 3]
    Pyz = stress_block[:, 4]
    Pzx = stress_block[:, 5]

    # Normal-stress difference components (traceless deviatoric)
    Pxx = stress_block[:, 0]
    Pyy = stress_block[:, 1]
    Pzz = stress_block[:, 2]
    Nxy = (Pxx - Pyy) / 2.0
    Nyz = (Pyy - Pzz) / 2.0

    # Compute autocorrelation for each component using FFT
    acf_xy = autocorr_fft(Pxy)
    acf_yz = autocorr_fft(Pyz)
    acf_zx = autocorr_fft(Pzx)
    acf_nxy = autocorr_fft(Nxy)
    acf_nyz = autocorr_fft(Nyz)

    # Average all 5 independent components for better statistics
    acf_avg = (acf_xy + acf_yz + acf_zx + acf_nxy + acf_nyz) / 5.0

    n_acf = len(acf_avg)

    # Apply window function if specified
    if window == 'hann':
        modified_hann = 0.5 * (1 + np.cos(np.pi * np.arange(n_acf) / (n_acf - 1)))
        acf_avg *= modified_hann

    # Time axis for ACF lags
    lags = np.arange(n_acf) * dt_s

    # Green-Kubo formula: η = (V / kBT) * ∫ SACF(t) dt
    visc_running = (V_m3 / (kB * T)) * cumtrapz(acf_avg, lags, initial=0.0)

    # Determine truncation index
    # Default: find the plateau region where the running integral stabilizes
    if truncation == 'tmax' and tmax_ps is not None:
        tmax_s = tmax_ps * 1e-12
        trunc_idx = min(np.argmin(np.abs(lags - tmax_s)), n_acf - 1)
    elif truncation == 'zero_cross':
        # First zero crossing of SACF
        nz = np.where(acf_avg[1:] <= 0)[0]
        trunc_idx = nz[0] + 1 if len(nz) > 0 else n_acf - 1
    elif truncation == 'first_min':
        # First minimum of running viscosity after initial rise
        search_start = max(1, int(0.05 * n_acf))
        trunc_idx = n_acf - 1
        for k in range(search_start, n_acf - 1):
            if visc_running[k] < visc_running[k - 1] and visc_running[k] < visc_running[k + 1]:
                trunc_idx = k
                break
    else:
        # Default: auto-detect plateau via first zero crossing of SACF
        # This is the most physically meaningful default — integrate only
        # while the SACF is positive (correlated), stop at decorrelation.
        nz = np.where(acf_avg[1:] <= 0)[0]
        if len(nz) > 0:
            trunc_idx = nz[0] + 1
        else:
            # SACF never crosses zero — use 10% of total length as safe limit
            trunc_idx = min(n_acf // 10, n_acf - 1)

    times = np.arange(len(visc_running)) * dt_s
    return times, visc_running, trunc_idx, acf_avg

# -----------------------------
# Block averaging and bootstrap
# -----------------------------

def block_split_array(arr: np.ndarray, n_blocks: int) -> List[np.ndarray]:
    n = arr.shape[0]
    bsz = n // n_blocks
    blocks = []
    for i in range(n_blocks):
        s = i * bsz
        e = (i + 1) * bsz if i < n_blocks - 1 else n
        blocks.append(arr[s:e])
    return blocks


def bootstrap_ci(samples: np.ndarray, ci: float = 0.95, n_bootstrap: int = 2000, seed: Optional[int] = 12345) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(samples)
    if n < 2:
        return float(samples[0]), float(samples[0])
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        means[i] = samples[idx].mean()
    alpha = (1.0 - ci) / 2.0
    low = np.quantile(means, alpha)
    high = np.quantile(means, 1.0 - alpha)
    return float(low), float(high)

# -----------------------------
# Elastic properties from stress fluctuations
# -----------------------------

def compute_pressure_statistics(stress_arr: np.ndarray) -> Dict[str, float]:
    """从应力时间序列计算有物理意义的静水压力统计量。

    Parameters
    ----------
    stress_arr : np.ndarray
        应力张量数组 (nsteps, 6), 顺序 [Pxx, Pyy, Pzz, Pxy, Pyz, Pzx], 单位 Pa

    Returns
    -------
    Dict[str, float]
        pressure_mean_GPa : 平均静水压力 P = (Pxx+Pyy+Pzz)/3
        pressure_std_GPa  : 静水压力涨落 (标准差)

    说明
    ----
    早期版本曾试图用 NVT 应力涨落 (K = kB·T/(V·Var(P)) 之类) 估计弹性模量,
    但完整的等温弹性常数 (Ray-Rahman 公式) 还需 Born 项 (势能二阶导) 和动能项,
    无法仅凭应力时间序列得到, 因此已移除 K/G/E/ν 输出。
    弹性常数请使用 SC-SS / SA-SS 应力-应变法。这里只保留物理意义明确的压力统计。
    """
    if stress_arr.size == 0 or len(stress_arr) < 2:
        return {
            'pressure_mean_GPa': float('nan'),
            'pressure_std_GPa': float('nan'),
        }

    P_hydro = (stress_arr[:, 0] + stress_arr[:, 1] + stress_arr[:, 2]) / 3.0
    return {
        'pressure_mean_GPa': float(np.mean(P_hydro) * 1e-9),
        'pressure_std_GPa': float(np.std(P_hydro, ddof=1) * 1e-9),
    }

# -----------------------------
# Plotting helpers with fallback
# -----------------------------

# Try to import plot modules, but don't fail if they're not available
try:
    from . import plot_functions
    from . import config_manager
    HAS_PLOT_MODULES = True
except ImportError:
    HAS_PLOT_MODULES = False
    print("Warning: plot_functions or config_manager not found. Using basic plotting fallback.")

def save_plot_rdf(r: np.ndarray, g_r: np.ndarray, out_png: str):
    """保存RDF图 - 使用统一的绘图配置或基础绘图"""
    if HAS_PLOT_MODULES:
        try:
            settings = config_manager.get_plot_settings()
            dpi = settings.get('dpi', 'standard')
            plot_functions.save_plot_rdf(r, g_r, out_png, dpi=dpi)
            return
        except Exception as e:
            print(f"Warning: Enhanced plotting failed ({e}), using basic plotting.")
    
    # Fallback to basic plotting
    if not HAS_MPL:
        print("Warning: Matplotlib not available, skipping RDF plot.")
        return
    plt.figure(figsize=(8, 6))
    plt.plot(r, g_r, linewidth=2, color='blue')
    plt.xlabel('r (Å)', fontsize=12)
    plt.ylabel('g(r)', fontsize=12)
    plt.title('Radial Distribution Function', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(T(f"[PLOT] RDF图像已保存到 {out_png}", f"[PLOT] RDF figure saved to {out_png}"))


def save_plot_msd(times: np.ndarray, msd: np.ndarray, D: float, out_png: str):
    """保存MSD图 - 使用统一的绘图配置或基础绘图"""
    if HAS_PLOT_MODULES:
        try:
            settings = config_manager.get_plot_settings()
            dpi = settings.get('dpi', 'standard')
            plot_functions.save_plot_msd(times, msd, D, out_png, dpi=dpi)
            return
        except Exception as e:
            print(f"Warning: Enhanced plotting failed ({e}), using basic plotting.")
    
    # Fallback to basic plotting
    if not HAS_MPL:
        print("Warning: Matplotlib not available, skipping MSD plot.")
        return
    plt.figure(figsize=(8, 6))
    plt.plot(times, msd, linewidth=2, color='red', marker='o', markersize=3, markevery=max(1, len(times)//20))
    if D > 0:
        fit = 6 * D * times
        plt.plot(times, fit, '--', color='black', linewidth=1.5, label=f'D = {D:.2e} m²/s')
        plt.legend()
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('MSD (m²)', fontsize=12)
    plt.title('Mean Square Displacement', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(T(f"[PLOT] MSD图像已保存到 {out_png}", f"[PLOT] MSD figure saved to {out_png}"))


def save_plot_msd_by_species(times: np.ndarray, msd_species: Dict[str, np.ndarray], out_png: str, species_diffusion: Dict = None):
    """保存按物种分类的MSD图 - 使用统一的绘图配置"""
    from .plot_functions import save_plot_msd_by_species as plot_msd_by_species_unified
    from .config_manager import get_plot_settings
    settings = get_plot_settings()
    dpi = settings.get('dpi', 'standard')
    
    # 转换 MSD 数据从 A^2 到 m^2
    msd_species_m2 = {sname: msd * (1e-10)**2 for sname, msd in msd_species.items()}
    
    # 转换 species_diffusion 格式：从 {species: {'D_m2_s': ...}} 到 {species: D_value}
    species_D = None
    if species_diffusion:
        species_D = {sname: data.get('D_m2_s', 0) for sname, data in species_diffusion.items()}
    
    plot_msd_by_species_unified(times, msd_species_m2, out_png, species_D, dpi=dpi)


def save_plot_vacf(times: np.ndarray, vacf: np.ndarray, out_png: str):
    """保存VACF图 - 使用统一的绘图配置"""
    from .plot_functions import save_plot_vacf as plot_vacf_unified
    from .config_manager import get_plot_settings
    settings = get_plot_settings()
    dpi = settings.get('dpi', 'standard')
    plot_vacf_unified(times, vacf, out_png, dpi=dpi)


def save_plot_molecular_friction_coefficients(species_diffusion: Dict, out_png: str):
    """保存分子摩擦系数图 - 使用统一的绘图配置"""
    from .plot_functions import save_plot_molecular_friction_coefficients as plot_friction_unified
    from .config_manager import get_plot_settings
    settings = get_plot_settings()
    dpi = settings.get('dpi', 'standard')
    plot_friction_unified(species_diffusion, out_png, dpi=dpi)


def save_plot_viscosity(times_list: List[np.ndarray], visc_list: List[np.ndarray], mean_curve: np.ndarray, out_png: str, labels: Optional[List[str]] = None):
    """保存粘度图 - 使用统一的绘图配置"""
    from .plot_functions import save_plot_viscosity as plot_viscosity_unified
    from .config_manager import get_plot_settings
    settings = get_plot_settings()
    dpi = settings.get('dpi', 'standard')
    plot_viscosity_unified(times_list, visc_list, mean_curve, out_png, labels, dpi=dpi)

def save_plot_energy_time(en_times: np.ndarray, energies: List[float], out_png: str, temps: Optional[List[float]] = None):
    """保存能量-时间图 - 使用统一的绘图配置"""
    from .plot_functions import save_plot_energy_time as plot_energy_unified
    from .config_manager import get_plot_settings
    settings = get_plot_settings()
    dpi = settings.get('dpi', 'standard')
    plot_energy_unified(en_times, energies, out_png, temps, dpi=dpi)


def save_plot_viscosity_from_sacf(data_file: str, out_png: str):
    """从SACF积分数据绘制粘度图 - 使用统一的绘图配置"""
    from .plot_functions import save_plot_viscosity_from_sacf as plot_sacf_unified
    from .config_manager import get_plot_settings
    settings = get_plot_settings()
    dpi = settings.get('dpi', 'standard')
    plot_sacf_unified(data_file, out_png, dpi=dpi)


def save_plot_pdos(frequencies_THz: np.ndarray, pdos: np.ndarray, out_png: str):
    """绘制 PDOS 图 - 使用统一的绘图配置"""
    from .plot_functions import save_plot_pdos as plot_pdos_unified
    from .config_manager import get_plot_settings
    settings = get_plot_settings()
    dpi = settings.get('dpi', 'standard')
    plot_pdos_unified(frequencies_THz, pdos, out_png, dpi=dpi)


def save_summary_table(summary_data: Dict, prefix: str):
    """Save a comprehensive summary table for Origin import"""
    summary_dat = f'{prefix}_summary_table.dat'
    
    # Create summary table with key results
    table_data = []
    headers = ['Property', 'Value', 'Unit', 'Method']
    
    # Add basic info
    table_data.append(['Temperature', f"{summary_data.get('T', 'N/A')}", 'K', 'Input'])
    table_data.append(['Frames', f"{summary_data.get('nframes', 'N/A')}", '-', 'XDATCAR'])
    table_data.append(['Atoms', f"{summary_data.get('natoms', 'N/A')}", '-', 'XDATCAR'])
    table_data.append(['Volume', f"{summary_data.get('volume_A3', 'N/A')}", 'A^3', 'Lattice'])
    if summary_data.get('density_g_cm3', 'N/A') != 'N/A':
        table_data.append(['Density', f"{summary_data['density_g_cm3']:.4f}", 'g/cm^3', 'Mass/Volume'])
    
    # Add RDF results
    if 'rdf_peak_r_A' in summary_data:
        table_data.append(['RDF_Peak_Position', f"{summary_data['rdf_peak_r_A']:.3f}", 'A', 'RDF'])
        table_data.append(['RDF_Peak_Height', f"{summary_data['rdf_peak_g']:.3f}", '-', 'RDF'])
    
    if 'coordination_number_to_first_min' in summary_data:
        table_data.append(['Coordination_Number', f"{summary_data['coordination_number_to_first_min']:.2f}", '-', 'RDF_Integration'])
    
    # Add diffusion results
    if 'diffusion_m2_s' in summary_data:
        table_data.append(['Diffusion_Coefficient', f"{summary_data['diffusion_m2_s']:.4e}", 'm^2/s', 'MSD_Fit'])
    
    # Add viscosity results
    if 'stokes_einstein_viscosity_Pa_s' in summary_data:
        table_data.append(['Viscosity_SE', f"{summary_data['stokes_einstein_viscosity_Pa_s']:.4e}", 'Pa*s', 'Stokes_Einstein'])
    
    if 'green_kubo_viscosity_Pa_s' in summary_data and summary_data['green_kubo_viscosity_Pa_s'] != 'N/A':
        table_data.append(['Viscosity_GK', f"{summary_data['green_kubo_viscosity_Pa_s']:.4e}", 'Pa*s', 'Green_Kubo'])
    
    # Add hydrostatic pressure statistics (物理意义明确; 弹性模量请用 SC-SS/SA-SS)
    if 'pressure_stats' in summary_data and summary_data['pressure_stats'] != 'N/A':
        ps = summary_data['pressure_stats']
        if not np.isnan(ps.get('pressure_mean_GPa', float('nan'))):
            table_data.append(['Mean_Pressure', f"{ps['pressure_mean_GPa']:.4f}", 'GPa', 'Stress_Average'])
        if not np.isnan(ps.get('pressure_std_GPa', float('nan'))):
            table_data.append(['Pressure_Std', f"{ps['pressure_std_GPa']:.4f}", 'GPa', 'Stress_Fluctuation'])
    
    # Save as tab-delimited file
    with open(summary_dat, 'w', encoding='utf-8') as f:
        f.write('\t'.join(headers) + '\n')
        for row in table_data:
            f.write('\t'.join(str(item) for item in row) + '\n')
    
    print(T(f"[OK] 汇总表格已保存到 {summary_dat}", f"[OK] Summary table saved to {summary_dat}"))


# -----------------------------
# Advanced Analysis Functions
# -----------------------------

def compute_partial_rdf(frac_coords: np.ndarray, lattice: np.ndarray, species: List[str], 
                       rmax: float = 10.0, nbins: int = 500) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute partial radial distribution functions for all species pairs"""
    nframes, natoms, _ = frac_coords.shape
    
    # Group atoms by species
    species_indices = {}
    for i, s in enumerate(species):
        if s not in species_indices:
            species_indices[s] = []
        species_indices[s].append(i)
    
    Vcell = abs(np.linalg.det(lattice))
    dr = rmax / nbins
    edges = np.linspace(0.0, rmax, nbins + 1)
    r = (edges[:-1] + edges[1:]) / 2.0
    
    partial_rdfs = {}
    
    # Compute all species pairs (including self-pairs)
    species_list = list(species_indices.keys())
    for i, species_a in enumerate(species_list):
        for j, species_b in enumerate(species_list):
            if i <= j:  # Avoid duplicate pairs (A-B same as B-A)
                pair_name = f"{species_a}-{species_b}"
                
                indices_a = species_indices[species_a]
                indices_b = species_indices[species_b]
                
                hist = np.zeros(nbins)
                
                for f in range(nframes):
                    pos = frac_coords[f]
                    
                    for idx_a in indices_a:
                        for idx_b in indices_b:
                            if species_a == species_b and idx_a >= idx_b:
                                continue  # Avoid double counting for same species
                            
                            # Calculate distance with PBC
                            diff_frac = pos[idx_a] - pos[idx_b]
                            diff_frac = diff_frac - np.rint(diff_frac)
                            diff_cart = diff_frac.dot(lattice)
                            r_dist = np.linalg.norm(diff_cart)
                            
                            if r_dist < rmax:
                                bin_idx = int(r_dist / dr)
                                if bin_idx < nbins:
                                    hist[bin_idx] += 1
                
                # Normalization
                N_a = len(indices_a)
                N_b = len(indices_b)
                
                if species_a == species_b:
                    total_pairs = N_a * (N_a - 1) / 2.0
                else:
                    total_pairs = N_a * N_b
                
                shell_vol = 4.0 * np.pi * r**2 * dr
                rho_b = N_b / Vcell

                g_r = np.zeros_like(hist, dtype=float)
                valid_indices = shell_vol > 1e-12
                # 期望对数 (理想气体): nframes · total_pairs · shell_vol / Vcell
                # (cross 对 total_pairs=N_a·N_b 时等价于 nframes·N_a·rho_b·shell_vol)
                # g(r) = 实测对数 / 期望对数; 与 compute_rdf 归一化一致, 大 r 处趋于 1。
                expected_pairs = nframes * total_pairs * shell_vol[valid_indices] / Vcell
                g_r[valid_indices] = hist[valid_indices] / expected_pairs
                
                partial_rdfs[pair_name] = (r, g_r)
    
    return partial_rdfs


def compute_structure_factor(frac_coords: np.ndarray, lattice: np.ndarray, 
                           qmax: float = 20.0, nq: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Compute static structure factor S(q)"""
    nframes, natoms, _ = frac_coords.shape
    
    # Generate q vectors
    q_values = np.linspace(0.1, qmax, nq)  # Start from 0.1 to avoid q=0
    
    # Convert to Cartesian coordinates
    cart_coords = np.zeros_like(frac_coords)
    for f in range(nframes):
        cart_coords[f] = frac_coords[f].dot(lattice)
    
    S_q = np.zeros(nq)
    
    for iq, q in enumerate(q_values):
        # For isotropic average, use q along z-direction
        q_vec = np.array([0, 0, q])
        
        S_sum = 0.0
        for f in range(nframes):
            positions = cart_coords[f]
            
            # Calculate structure factor for this frame
            cos_sum = 0.0
            sin_sum = 0.0
            
            for i in range(natoms):
                q_dot_r = np.dot(q_vec, positions[i])
                cos_sum += np.cos(q_dot_r)
                sin_sum += np.sin(q_dot_r)
            
            S_frame = (cos_sum**2 + sin_sum**2) / natoms
            S_sum += S_frame
        
        S_q[iq] = S_sum / nframes
    
    return q_values, S_q


def compute_partial_coordination_numbers(frac_coords: np.ndarray, lattice: np.ndarray, 
                                       species: List[str], r_cutoffs: Dict[str, float] = None) -> Dict[str, float]:
    """Compute partial coordination numbers for each species pair"""
    if r_cutoffs is None:
        # Default cutoffs (can be refined based on RDF minima)
        r_cutoffs = {
            'Ga-Ga': 3.5, 'Ga-In': 3.5, 'Ga-Sn': 3.5,
            'In-In': 3.5, 'In-Sn': 3.5, 'Sn-Sn': 3.5
        }
    
    nframes, natoms, _ = frac_coords.shape
    
    # Group atoms by species
    species_indices = {}
    for i, s in enumerate(species):
        if s not in species_indices:
            species_indices[s] = []
        species_indices[s].append(i)
    
    coordination_numbers = {}
    
    # Calculate for each species pair
    species_list = list(species_indices.keys())
    for species_a in species_list:
        for species_b in species_list:
            pair_name = f"{species_a}-{species_b}"
            
            if pair_name not in r_cutoffs:
                continue
            
            r_cut = r_cutoffs[pair_name]
            indices_a = species_indices[species_a]
            indices_b = species_indices[species_b]
            
            total_cn = 0.0
            count = 0
            
            for f in range(nframes):
                pos = frac_coords[f]
                
                for idx_a in indices_a:
                    cn_a = 0
                    for idx_b in indices_b:
                        if idx_a == idx_b:
                            continue
                        
                        # Calculate distance with PBC
                        diff_frac = pos[idx_a] - pos[idx_b]
                        diff_frac = diff_frac - np.rint(diff_frac)
                        diff_cart = diff_frac.dot(lattice)
                        r_dist = np.linalg.norm(diff_cart)
                        
                        if r_dist <= r_cut:
                            cn_a += 1
                    
                    total_cn += cn_a
                    count += 1
            
            coordination_numbers[pair_name] = total_cn / count if count > 0 else 0.0
    
    return coordination_numbers


def compute_msd_higher_moments(frac_coords: np.ndarray, lattice: np.ndarray, dt_fs: float) -> Dict[str, np.ndarray]:
    """Compute higher-order moments of MSD for dynamic heterogeneity analysis"""
    nframes, natoms, _ = frac_coords.shape
    
    # Unwrap positions
    unwrapped = unwrap_positions(frac_coords)
    cart = fractional_to_cart(unwrapped, lattice)
    
    # Calculate squared displacements for each atom and time lag
    times = np.arange(nframes) * dt_fs * fs2s
    
    # Store displacement data for statistical analysis
    max_tau = min(nframes // 4, 1000)  # Limit for computational efficiency
    
    msd_moments = {
        'times': times[:max_tau],
        'msd_mean': np.zeros(max_tau),
        'msd_std': np.zeros(max_tau),
        'msd_skewness': np.zeros(max_tau),
        'msd_kurtosis': np.zeros(max_tau),
        'alpha2': np.zeros(max_tau),  # Non-Gaussian parameter
    }
    
    for tau in range(1, max_tau):
        if tau >= nframes:
            break
            
        # Calculate squared displacements for all atoms at this time lag
        displacements = cart[tau:] - cart[:-tau]  # Shape: (nframes-tau, natoms, 3)
        squared_displacements = np.sum(displacements**2, axis=2)  # Shape: (nframes-tau, natoms)
        
        # Flatten to get all displacement values
        all_displacements = squared_displacements.flatten()
        
        if len(all_displacements) > 0:
            # Calculate moments
            mean_msd = np.mean(all_displacements)
            std_msd = np.std(all_displacements)
            
            msd_moments['msd_mean'][tau] = mean_msd
            msd_moments['msd_std'][tau] = std_msd
            
            # Higher moments (normalized)
            if std_msd > 0:
                normalized = (all_displacements - mean_msd) / std_msd
                msd_moments['msd_skewness'][tau] = np.mean(normalized**3)
                msd_moments['msd_kurtosis'][tau] = np.mean(normalized**4) - 3  # Excess kurtosis
                
                # Non-Gaussian parameter alpha2 = <r^4>/(3<r^2>^2) - 1
                mean_r4 = np.mean(all_displacements**2)
                mean_r2_squared = mean_msd**2
                if mean_r2_squared > 0:
                    msd_moments['alpha2'][tau] = mean_r4 / (3 * mean_r2_squared) - 1
    
    return msd_moments


def compute_bond_length_distribution(frac_coords: np.ndarray, lattice: np.ndarray, 
                                   species: List[str], r_max: float = 4.0, nbins: int = 200) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute bond length distributions for different species pairs"""
    nframes, natoms, _ = frac_coords.shape
    
    # Group atoms by species
    species_indices = {}
    for i, s in enumerate(species):
        if s not in species_indices:
            species_indices[s] = []
        species_indices[s].append(i)
    
    bond_distributions = {}
    edges = np.linspace(0, r_max, nbins + 1)
    r_centers = (edges[:-1] + edges[1:]) / 2.0
    
    # Calculate for each species pair
    species_list = list(species_indices.keys())
    for i, species_a in enumerate(species_list):
        for j, species_b in enumerate(species_list):
            if i <= j:  # Avoid duplicate pairs
                pair_name = f"{species_a}-{species_b}"
                
                indices_a = species_indices[species_a]
                indices_b = species_indices[species_b]
                
                all_distances = []
                
                for f in range(nframes):
                    pos = frac_coords[f]
                    
                    for idx_a in indices_a:
                        for idx_b in indices_b:
                            if species_a == species_b and idx_a >= idx_b:
                                continue
                            
                            # Calculate distance with PBC
                            diff_frac = pos[idx_a] - pos[idx_b]
                            diff_frac = diff_frac - np.rint(diff_frac)
                            diff_cart = diff_frac.dot(lattice)
                            r_dist = np.linalg.norm(diff_cart)
                            
                            if r_dist <= r_max:
                                all_distances.append(r_dist)
                
                if all_distances:
                    hist, _ = np.histogram(all_distances, bins=edges, density=True)
                    bond_distributions[pair_name] = (r_centers, hist)
    
    return bond_distributions


def compute_bond_angle_distribution(frac_coords: np.ndarray, lattice: np.ndarray, 
                                  species: List[str], r_cutoff: float = 3.5, nbins: int = 180) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute bond angle distributions for triplets of atoms"""
    nframes, natoms, _ = frac_coords.shape
    
    # Group atoms by species
    species_indices = {}
    for i, s in enumerate(species):
        if s not in species_indices:
            species_indices[s] = []
        species_indices[s].append(i)
    
    angle_distributions = {}
    edges = np.linspace(0, 180, nbins + 1)
    angle_centers = (edges[:-1] + edges[1:]) / 2.0
    
    # Calculate for each central atom species
    for central_species in species_indices.keys():
        central_indices = species_indices[central_species]
        
        all_angles = []
        
        for f in range(nframes):
            pos = frac_coords[f]
            
            for central_idx in central_indices:
                central_pos = pos[central_idx]
                
                # Find neighbors within cutoff
                neighbors = []
                for i in range(natoms):
                    if i == central_idx:
                        continue
                    
                    diff_frac = central_pos - pos[i]
                    diff_frac = diff_frac - np.rint(diff_frac)
                    diff_cart = diff_frac.dot(lattice)
                    r_dist = np.linalg.norm(diff_cart)
                    
                    if r_dist <= r_cutoff:
                        neighbors.append((i, diff_cart))
                
                # Calculate angles between all neighbor pairs
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        vec1 = neighbors[i][1]
                        vec2 = neighbors[j][1]
                        
                        # Calculate angle
                        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
                        angle = np.arccos(cos_angle) * 180.0 / np.pi
                        
                        all_angles.append(angle)
        
        if all_angles:
            hist, _ = np.histogram(all_angles, bins=edges, density=True)
            angle_distributions[central_species] = (angle_centers, hist)
    
    return angle_distributions


def save_advanced_analysis_results(prefix: str, partial_rdfs: Dict, structure_factor: Tuple, 
                                 coordination_numbers: Dict, msd_moments: Dict, 
                                 bond_lengths: Dict, bond_angles: Dict):
    """Save all advanced analysis results to Origin-compatible files"""
    
    # Save partial RDFs
    for pair_name, (r, g_r) in partial_rdfs.items():
        filename = f"{prefix}_partial_rdf_{pair_name.replace('-', '_')}.dat"
        np.savetxt(filename, np.column_stack((r, g_r)), 
                  header=f'r_A\tg_{pair_name.replace("-", "_")}_r', 
                  comments='', delimiter='\t')
        print(T(f"[OK] 偏RDF ({pair_name}) 已保存到 {filename}", f"[OK] Partial RDF ({pair_name}) saved to {filename}"))
    
    # Save structure factor
    if structure_factor:
        q_values, S_q = structure_factor
        filename = f"{prefix}_structure_factor.dat"
        np.savetxt(filename, np.column_stack((q_values, S_q)), 
                  header='q_inv_A\tS_q', comments='', delimiter='\t')
        print(T(f"[OK] 结构因子 S(q) 已保存到 {filename}", f"[OK] Structure factor S(q) saved to {filename}"))
    
    # Save coordination numbers
    if coordination_numbers:
        filename = f"{prefix}_coordination_numbers.dat"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('Pair\tCoordination_Number\n')
            for pair, cn in coordination_numbers.items():
                f.write(f'{pair}\t{cn:.3f}\n')
        print(T(f"[OK] 配位数已保存到 {filename}", f"[OK] Coordination numbers saved to {filename}"))
    
    # Save MSD moments
    if msd_moments:
        filename = f"{prefix}_msd_moments.dat"
        data_cols = [
            msd_moments['times'] * 1e12,  # Convert to ps
            msd_moments['msd_mean'],
            msd_moments['msd_std'],
            msd_moments['msd_skewness'],
            msd_moments['msd_kurtosis'],
            msd_moments['alpha2']
        ]
        header = 'time_ps\tMSD_mean_m2\tMSD_std_m2\tMSD_skewness\tMSD_kurtosis\talpha2'
        np.savetxt(filename, np.column_stack(data_cols), 
                  header=header, comments='', delimiter='\t')
        print(T(f"[OK] MSD高阶矩已保存到 {filename}", f"[OK] MSD higher moments saved to {filename}"))
    
    # Save bond length distributions
    for pair_name, (r, hist) in bond_lengths.items():
        filename = f"{prefix}_bond_length_{pair_name.replace('-', '_')}.dat"
        np.savetxt(filename, np.column_stack((r, hist)), 
                  header=f'r_A\tP_{pair_name.replace("-", "_")}', 
                  comments='', delimiter='\t')
        print(T(f"[OK] 键长分布 ({pair_name}) 已保存到 {filename}", f"[OK] Bond-length distribution ({pair_name}) saved to {filename}"))
    
    # Save bond angle distributions
    for species, (angles, hist) in bond_angles.items():
        filename = f"{prefix}_bond_angle_{species}.dat"
        np.savetxt(filename, np.column_stack((angles, hist)), 
                  header=f'angle_deg\tP_{species}', 
                  comments='', delimiter='\t')
        print(T(f"[OK] 键角分布 ({species}) 已保存到 {filename}", f"[OK] Bond-angle distribution ({species}) saved to {filename}"))


def plot_advanced_analysis_results(prefix: str, partial_rdfs: Dict, structure_factor: Tuple, 
                                 msd_moments: Dict, bond_lengths: Dict, bond_angles: Dict):
    """Generate plots for advanced analysis results"""
    if not HAS_MPL:
        print("Warning: Matplotlib not available, skipping advanced analysis plots")
        return
    
    # Plot partial RDFs
    if partial_rdfs:
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, len(partial_rdfs)))
        
        for i, (pair_name, (r, g_r)) in enumerate(partial_rdfs.items()):
            plt.plot(r, g_r, label=f'g_{{{pair_name}}}(r)', color=colors[i], linewidth=2)
        
        plt.xlabel('r (A)')
        plt.ylabel('g(r)')
        plt.title('Partial Radial Distribution Functions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{prefix}_partial_rdfs.png", dpi=200)
        plt.close()
        print(T(f"[PLOT] 偏RDF图已保存到 {prefix}_partial_rdfs.png", f"[PLOT] Partial-RDF figure saved to {prefix}_partial_rdfs.png"))
    
    # Plot structure factor
    if structure_factor:
        q_values, S_q = structure_factor
        plt.figure(figsize=(8, 6))
        plt.plot(q_values, S_q, 'b-', linewidth=2)
        plt.xlabel('q (A^-1)')
        plt.ylabel('S(q)')
        plt.title('Static Structure Factor')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{prefix}_structure_factor.png", dpi=200)
        plt.close()
        print(T(f"[PLOT] 结构因子图已保存到 {prefix}_structure_factor.png", f"[PLOT] Structure-factor figure saved to {prefix}_structure_factor.png"))
    
    # Plot MSD moments
    if msd_moments:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        times_ps = msd_moments['times'] * 1e12
        
        # MSD mean and std
        axes[0,0].plot(times_ps, msd_moments['msd_mean'], 'b-', label='Mean')
        axes[0,0].fill_between(times_ps, 
                              msd_moments['msd_mean'] - msd_moments['msd_std'],
                              msd_moments['msd_mean'] + msd_moments['msd_std'],
                              alpha=0.3, label='+/-1sigma')
        axes[0,0].set_xlabel('Time (ps)')
        axes[0,0].set_ylabel('MSD (m^2)')
        axes[0,0].set_title('MSD Statistics')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Skewness
        axes[0,1].plot(times_ps, msd_moments['msd_skewness'], 'g-', linewidth=2)
        axes[0,1].set_xlabel('Time (ps)')
        axes[0,1].set_ylabel('Skewness')
        axes[0,1].set_title('MSD Skewness')
        axes[0,1].grid(True, alpha=0.3)
        
        # Kurtosis
        axes[1,0].plot(times_ps, msd_moments['msd_kurtosis'], 'r-', linewidth=2)
        axes[1,0].set_xlabel('Time (ps)')
        axes[1,0].set_ylabel('Excess Kurtosis')
        axes[1,0].set_title('MSD Kurtosis')
        axes[1,0].grid(True, alpha=0.3)
        
        # Non-Gaussian parameter
        axes[1,1].plot(times_ps, msd_moments['alpha2'], 'm-', linewidth=2)
        axes[1,1].set_xlabel('Time (ps)')
        axes[1,1].set_ylabel('alpha2')
        axes[1,1].set_title('Non-Gaussian Parameter')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{prefix}_msd_moments.png", dpi=200)
        plt.close()
        print(T(f"[PLOT] MSD高阶矩图已保存到 {prefix}_msd_moments.png", f"[PLOT] MSD higher-moment figure saved to {prefix}_msd_moments.png"))
    
    # Plot bond length distributions
    if bond_lengths:
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(bond_lengths)))
        
        for i, (pair_name, (r, hist)) in enumerate(bond_lengths.items()):
            plt.plot(r, hist, label=f'{pair_name}', color=colors[i], linewidth=2)
        
        plt.xlabel('Bond Length (A)')
        plt.ylabel('Probability Density')
        plt.title('Bond Length Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{prefix}_bond_lengths.png", dpi=200)
        plt.close()
        print(T(f"[PLOT] 键长分布图已保存到 {prefix}_bond_lengths.png", f"[PLOT] Bond-length figure saved to {prefix}_bond_lengths.png"))
    
    # Plot bond angle distributions
    if bond_angles:
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(bond_angles)))
        
        for i, (species, (angles, hist)) in enumerate(bond_angles.items()):
            plt.plot(angles, hist, label=f'{species} center', color=colors[i], linewidth=2)
        
        plt.xlabel('Bond Angle (°)')
        plt.ylabel('Probability Density')
        plt.title('Bond Angle Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{prefix}_bond_angles.png", dpi=200)
        plt.close()
        print(T(f"[PLOT] 键角分布图已保存到 {prefix}_bond_angles.png", f"[PLOT] Bond-angle figure saved to {prefix}_bond_angles.png"))


def print_output_summary(prefix: str):
    """Print a summary of all generated output files"""
    print(f"\n{'='*60}")
    print(T(f"[FILES] 输出文件汇总 (前缀: {prefix})", f"[FILES] Output file summary (prefix: {prefix})"))
    print(f"{'='*60}")
    
    # Data files (.dat format for Origin)
    print(T("[DATA] 基础数据文件 (.dat格式，Origin可直接导入):", "[DATA] Basic data files (.datformat, Origin-importable):"))
    basic_data_files = [
        (f"{prefix}_rdf.dat", "径向分布函数 g(r)"),
        (f"{prefix}_msd.dat", "均方位移 MSD"),
        (f"{prefix}_msd_by_species.dat", "各元素的MSD"),
        (f"{prefix}_vacf.dat", "速度自相关函数 VACF"),
        (f"{prefix}_pdos.dat", "声子态密度 PDOS (从VACF计算)"),
        (f"{prefix}_stress.dat", "应力张量时间序列"),
        (f"{prefix}_acf.dat", "应力自相关函数 ACF"),
        (f"{prefix}_sacf.dat", "平均应力自相关函数 SACF"),
        (f"{prefix}_visc_running.dat", "Green-Kubo运行粘度"),
        (f"{prefix}_viscosity_from_sacf.dat", "SACF积分粘度"),
        (f"{prefix}_molecular_friction.dat", "分子摩擦系数"),
        (f"{prefix}_friction.dat", "宏观摩擦系数"),
        (f"{prefix}_energy.dat", "能量-温度时间序列"),
        (f"{prefix}_summary_table.dat", "结果汇总表格")
    ]
    
    for filename, description in basic_data_files:
        if os.path.exists(filename):
            print(f"  [OK] {filename:<35} - {description}")
        else:
            print(T(f"  [--] {filename:<35} - {description} (未生成)", f"  [--] {filename:<35} - {description} (not generated)"))
    
    # Advanced analysis data files
    print(T(f"\n[ADV] 高级分析数据文件:", f"\n[ADV] Advanced analysis data files:"))
    advanced_data_files = [
        (f"{prefix}_structure_factor.dat", "静态结构因子 S(q)"),
        (f"{prefix}_coordination_numbers.dat", "偏配位数"),
        (f"{prefix}_msd_moments.dat", "MSD高阶矩 (动力学非均匀性)"),
    ]
    
    # Check for partial RDF files (dynamic based on species)
    import glob
    partial_rdf_files = glob.glob(f"{prefix}_partial_rdf_*.dat")
    bond_length_files = glob.glob(f"{prefix}_bond_length_*.dat")
    bond_angle_files = glob.glob(f"{prefix}_bond_angle_*.dat")
    
    for filename, description in advanced_data_files:
        if os.path.exists(filename):
            print(f"  [OK] {filename:<35} - {description}")
        else:
            print(T(f"  [--] {filename:<35} - {description} (未生成)", f"  [--] {filename:<35} - {description} (not generated)"))
    
    if partial_rdf_files:
        print(f"  [OK] 偏RDF文件 ({len(partial_rdf_files)}个)      - 各元素对的径向分布函数")
    if bond_length_files:
        print(f"  [OK] 键长分布文件 ({len(bond_length_files)}个)    - 各元素对的键长分布")
    if bond_angle_files:
        print(f"  [OK] 键角分布文件 ({len(bond_angle_files)}个)    - 各元素的键角分布")
    
    # Image files (.png format)
    print(f"\n[IMG] 基础图像文件 (.png格式):")
    basic_image_files = [
        (f"{prefix}_rdf.png", "RDF图"),
        (f"{prefix}_msd.png", "MSD图"),
        (f"{prefix}_msd_by_species.png", "各元素MSD对比图"),
        (f"{prefix}_vacf.png", "VACF图"),
        (f"{prefix}_pdos.png", "声子态密度图 (从VACF计算)"),
        (f"{prefix}_viscosity.png", "粘度对比图"),
        (f"{prefix}_sacf.png", "SACF图"),
        (f"{prefix}_viscosity_comparison.png", "粘度方法对比图"),
        (f"{prefix}_viscosity_from_sacf.png", "SACF积分粘度图"),
        (f"{prefix}_molecular_friction.png", "分子摩擦系数图"),
        (f"{prefix}_friction.png", "宏观摩擦系数图"),
        (f"{prefix}_energy.png", "能量-温度图")
    ]
    
    for filename, description in basic_image_files:
        if os.path.exists(filename):
            print(f"  [OK] {filename:<40} - {description}")
        else:
            print(T(f"  [--] {filename:<40} - {description} (未生成)", f"  [--] {filename:<40} - {description} (not generated)"))
    
    # Advanced analysis image files
    print(T(f"\n[ADV] 高级分析图像文件:", f"\n[ADV] Advanced analysis image files:"))
    advanced_image_files = [
        (f"{prefix}_partial_rdfs.png", "偏RDF对比图"),
        (f"{prefix}_structure_factor.png", "静态结构因子图"),
        (f"{prefix}_msd_moments.png", "MSD高阶矩图"),
        (f"{prefix}_bond_lengths.png", "键长分布图"),
        (f"{prefix}_bond_angles.png", "键角分布图")
    ]
    
    for filename, description in advanced_image_files:
        if os.path.exists(filename):
            print(f"  [OK] {filename:<40} - {description}")
        else:
            print(T(f"  [--] {filename:<40} - {description} (未生成)", f"  [--] {filename:<40} - {description} (not generated)"))
    
    # Summary files
    print(T(f"\n[SUM] 摘要文件:", f"\n[SUM] Summary files:"))
    summary_files = [
        (f"{prefix}_summary.json", "完整结果摘要 (JSON格式)"),
        (f"{prefix}_summary_table.dat", "关键结果表格 (Origin格式)")
    ]
    
    for filename, description in summary_files:
        if os.path.exists(filename):
            print(f"  [OK] {filename:<35} - {description}")
        else:
            print(T(f"  [--] {filename:<35} - {description} (未生成)", f"  [--] {filename:<35} - {description} (not generated)"))
    
    print(T(f"\n[TIP] 使用建议:", f"\n[TIP] Usage tips:"))
    print(T(f"  - 所有 .dat 文件可直接导入 Origin 进行分析和绘图", f"  - All .dat files can be imported directly into Origin for analysis and plotting"))
    print(T(f"  - .png 文件提供快速预览，可用于报告和演示", f"  - .png files give quick previews for reports and presentations"))
    print(T(f"  - summary_table.dat 包含所有关键结果，便于对比分析", f"  - summary_table.dat contains all key results for comparison"))
    print(f"{'='*60}")


# -----------------------------
# Friction calculation function (from viscosity_to_friction.py)
# -----------------------------

def calc_friction(eta: float, U_list: List[float], p: float, h: float, out_prefix: str = "friction_result"):
    """
    根据粘度 eta 计算宏观摩擦系数 μ (润滑理论)
    μ = eta * U / (p * h)
    
    参数:
        eta: float, 粘度 (Pa*s)
        U_list: list or np.ndarray, 相对速度数组 (m/s)
        p: float, 载荷压力 (Pa)
        h: float, 润滑膜厚度 (m)
        out_prefix: str, 输出文件名前缀
    
    注意: 这是宏观摩擦系数mu，不同于分子摩擦系数xi
    """
    U = np.array(U_list)
    mu = eta * U / (p * h)

    # 保存表格为Origin友好的DAT格式
    dat_name = f"{out_prefix}.dat"
    np.savetxt(dat_name, np.column_stack((U, mu)), 
               header='Velocity_m_s\tFrictionCoeff_mu', 
               comments='', delimiter='\t')
    print(T(f"[OK] 宏观摩擦系数结果已保存到 {dat_name}", f"[OK] Macroscopic-friction results saved to {dat_name}"))

    # 绘制曲线
    if HAS_MPL:
        plt.figure(figsize=(6,5))
        plt.plot(U, mu, "o-", lw=2, label=f"eta={eta:.3e} Pa*s, p={p/1e6:.2f} MPa, h={h*1e9:.1f} nm")
        plt.xlabel("Velocity U (m/s)")
        plt.ylabel("Macroscopic Friction coefficient mu")
        plt.title("mu vs U (Lubrication Theory: mu = eta*U/p*h)")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.6)
        fig_name = f"{out_prefix}.png"
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(T(f"[OK] 宏观摩擦系数图像已保存到 {fig_name}", f"[OK] Macroscopic-friction figure saved to {fig_name}"))

# -----------------------------
# Main driver
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser(description='Comprehensive VASP AIMD post-processing for viscosity (GK & SE) and friction.')
    p.add_argument('--xdatcar', type=str, default='XDATCAR', help='XDATCAR path (fractional coordinates)')
    p.add_argument('--poscar', type=str, default='POSCAR', help='POSCAR path (to get species & counts)')
    p.add_argument('--outcar', type=str, default='OUTCAR', help='OUTCAR path')
    p.add_argument('--vasprun', type=str, default='vasprun.xml', help='vasprun.xml path (optional)')
    p.add_argument('--oszicar', type=str, default='OSZICAR', help='OSZICAR path (optional) for energy/time parsing')
    p.add_argument('--T', type=float, default=300.0, help='Temperature in K')
    p.add_argument('--potim', type=float, default=1.0, help='POTIM (fs) time step')
    p.add_argument('--r_eff', type=float, default=None, help='Effective hydrodynamic radius in A (if none, use RDF_peak/2)')
    p.add_argument('--n_blocks', type=int, default=5, help='Number of blocks for block averaging')
    p.add_argument('--stable_frac', type=float, default=0.3, help='Fraction of trajectory to discard as transient')
    p.add_argument('--fit_start_frac', type=float, default=0.2, help='MSD fit start fraction')
    p.add_argument('--fit_end_frac', type=float, default=0.8, help='MSD fit end fraction')
    p.add_argument('--truncation', type=str, choices=[None, 'tmax', 'zero_cross', 'first_min'], default=None, help='GK truncation rule')
    p.add_argument('--tmax_ps', type=float, default=None, help='If truncation=tmax, integrate up to this [ps]')
    p.add_argument('--window', type=str, choices=[None, 'hann'], default=None, help='Window for ACF before integration')
    p.add_argument('--bootstrap', action='store_true', help='Compute bootstrap CI for viscosity')
    p.add_argument('--n_bootstrap', type=int, default=2000, help='Bootstrap resamples')
    p.add_argument('--ci', type=float, default=0.95, help='Bootstrap CI level')
    p.add_argument('--out_prefix', type=str, default='aimd_postproc', help='Output prefix')
    p.add_argument('--no_plots', action='store_true', help='Do not generate PNG plots')
    p.add_argument('--no_intermediates', action='store_true', help='Do not save intermediate files (stress, ACF, running viscosity)')
    p.add_argument('--save_intermediates', action='store_true', help='[DEPRECATED] Use --no_intermediates to disable. Intermediate files now saved by default.')
    
    # Arguments for friction calculation
    p.add_argument('--friction_U', type=str, help='Space-separated list of velocities (m/s) for friction calculation. E.g., "0.01 0.1 1.0"')
    p.add_argument('--friction_p', type=float, help='Load pressure (Pa) for friction calculation')
    p.add_argument('--friction_h', type=float, help='Lubricant film thickness (m) for friction calculation')
    
    # Arguments for advanced analysis
    p.add_argument('--advanced_analysis', action='store_true', help='Perform advanced structural and dynamical analysis')
    p.add_argument('--partial_rdf', action='store_true', help='Compute partial RDFs for all species pairs')
    p.add_argument('--structure_factor', action='store_true', help='Compute static structure factor S(q)')
    p.add_argument('--coordination_analysis', action='store_true', help='Compute partial coordination numbers')
    p.add_argument('--msd_moments', action='store_true', help='Compute higher-order MSD moments for dynamic heterogeneity')
    p.add_argument('--bond_analysis', action='store_true', help='Compute bond length and angle distributions')
    p.add_argument('--qmax', type=float, default=20.0, help='Maximum q value for structure factor (A^-1)')
    p.add_argument('--bond_cutoff', type=float, default=3.5, help='Cutoff distance for bond analysis (A)')

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("\n--- AIMD Post-processing Script Started ---")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Output prefix: {args.out_prefix}")
    print(f"Temperature: {args.T} K, POTIM: {args.potim} fs")

    # Basic checks
    have_xdatcar = os.path.isfile(args.xdatcar)
    have_outcar = os.path.isfile(args.outcar)
    have_vasprun = os.path.isfile(args.vasprun)
    have_poscar = os.path.isfile(args.poscar)
    have_oszicar = os.path.isfile(args.oszicar) if args.oszicar else False

    if not have_xdatcar:
        print(f'Error: XDATCAR file not found at {args.xdatcar}. RDF/MSD analysis requires XDATCAR. Exiting.')
        sys.exit(1)
    print(f'[OK] XDATCAR found: {args.xdatcar}')
    print(f'Reading XDATCAR: {args.xdatcar} ...')
    frac_coords, lattice, elems, counts = read_xdatcar_frames(args.xdatcar)
    nframes, natoms, _ = frac_coords.shape
    print(f'Loaded {nframes} frames with {natoms} atoms from XDATCAR.')
    print(f'Lattice vectors (Angstrom):\n{lattice}')

    # Try to read POSCAR for species if present
    species = []
    if have_poscar:
        try:
            elems_p, counts_p, scale_p, lattice_p = read_poscar_header(args.poscar)
            if elems_p and counts_p:
                species = []
                for e, c in zip(elems_p, counts_p):
                    species.extend([e] * c)
            print(f'[OK] POSCAR found and parsed: {args.poscar}. Species: {elems_p}, Counts: {counts_p}')
        except Exception as e:
            print(f'Warning: Could not parse POSCAR {args.poscar}: {e}. Attempting to infer species from XDATCAR header.')
            species = []
    else:
        print(f'Info: POSCAR file not found at {args.poscar}. Will attempt to infer species from XDATCAR header.')

    # fallback: if elems and counts from XDATCAR header available
    if len(species) == 0 and len(elems) > 0 and len(counts) > 0:
        species = []
        for e, c in zip(elems, counts):
            species.extend([e] * c)
        print(f'Inferred species from XDATCAR header: {elems}, Counts: {counts}')

    # Convert lattice units: we assume lattice in A already from parser
    V_A3 = abs(np.linalg.det(lattice))
    V_m3 = V_A3 * A32m3
    print(f'Calculated cell volume: {V_A3:.3f} A^3 ({V_m3:.3e} m^3)')

    # 计算密度 (g/cm³)
    density_g_cm3 = float('nan')
    if len(species) == natoms:
        total_mass_amu = sum(ATOMIC_MASSES.get(s, 0.0) for s in species)
        if total_mass_amu > 0:
            # 1 amu = 1.66054e-24 g, 1 Å³ = 1e-24 cm³
            density_g_cm3 = total_mass_amu * 1.66054e-24 / (V_A3 * 1e-24)
            print(f'Calculated density: {density_g_cm3:.4f} g/cm³ (total mass: {total_mass_amu:.2f} amu)')
        else:
            print('Warning: Could not calculate density — unknown atomic masses.')
    else:
        print('Info: Species not available, skipping density calculation.')

    # Compute RDF (use subset of frames for performance)
    print('\n--- Computing Radial Distribution Function (RDF) ---')
    rdf_stride = max(1, nframes // 1000)  # Use at most 1000 frames for RDF
    frac_coords_rdf = frac_coords[::rdf_stride]
    print(f'Using {frac_coords_rdf.shape[0]} frames (every {rdf_stride}th frame) for RDF calculation.')
    r, g_r = compute_rdf(frac_coords_rdf, lattice, rmax=min(8.0, np.max(np.linalg.norm(lattice, axis=1))/3.0), nbins=200)
    peak_r, peak_g = find_rdf_first_peak(r, g_r)
    print(f'RDF first peak at r = {peak_r:.3f} A, g(r) = {peak_g:.3f}')
    
    # Compute coordination number
    Vcell = abs(np.linalg.det(lattice))
    rho = natoms / Vcell  # number density in atoms/A^3
    
    # Find first minimum after peak for coordination cutoff
    first_min_r = find_rdf_first_minimum(r, g_r, peak_r)
    coordination_number = compute_coordination_number(r, g_r, rho, first_min_r)
    
    print(f'First minimum at r = {first_min_r:.3f} A')
    print(f'Coordination number (up to first minimum): {coordination_number:.2f}')
    
    # Also compute coordination number up to peak
    cn_to_peak = compute_coordination_number(r, g_r, rho, peak_r)
    print(f'Coordination number (up to first peak): {cn_to_peak:.2f}')
    
    # Check g(r) normalization at large r
    large_r_indices = r > 0.8 * np.max(r)
    if np.any(large_r_indices):
        g_r_large = np.mean(g_r[large_r_indices])
        print(f'g(r) at large r (should be ~1.0): {g_r_large:.3f}')

    # decide effective radius
    # Stokes-Einstein: η = kBT / (6πRD)
    # R 是流体力学半径 ≈ 最近邻距离/2 (RDF第一峰位置是最近邻距离，约等于粒子直径)
    if args.r_eff is None:
        r_eff_A = peak_r / 2.0
        print(f'Using R_eff = RDF_peak/2 (hydrodynamic radius): {r_eff_A:.3f} A  (peak_r = {peak_r:.3f} A)')
    else:
        r_eff_A = float(args.r_eff)
        print(f'Using user-supplied R_eff: {r_eff_A:.3f} A')
    r_eff_m = r_eff_A * 1e-10

    # MSD and D (use subset of frames for performance)
    print('\n--- Computing Mean Squared Displacement (MSD) ---')
    msd_stride = max(1, nframes // 5000)  # Use at most 5000 frames for MSD
    frac_coords_msd = frac_coords[::msd_stride]
    print(f'Using {frac_coords_msd.shape[0]} frames (every {msd_stride}th frame) for MSD calculation.')
    times_msd, msd = compute_msd(frac_coords_msd, lattice, args.potim * msd_stride)
    # convert msd from A^2 to m^2: note compute_msd returns in A^2 because lattice in A; convert
    msd_m2 = msd * (1e-10)**2
    D, slope, intercept = estimate_diffusion_from_msd(times_msd, msd_m2, fit_start_frac=args.fit_start_frac, fit_end_frac=args.fit_end_frac)
    print(f'Estimated diffusion D = {D:.4e} m^2/s from MSD linear fit (slope={slope:.4e})')

    # Additionally compute MSD per species if species info available
    msd_species = None
    species_diffusion = {}
    if len(species) == natoms:
        try:
            print('Computing MSD separately for each species...')
            times_species_msd, msd_species_dict = compute_msd_by_species(frac_coords_msd, lattice, species, args.potim * msd_stride)
            msd_species = msd_species_dict
            print(f'Computed MSD for species: {list(msd_species.keys())}')
            
            # Fit diffusion coefficients for each species
            species_diffusion = {}
            print('\n--- Diffusion Coefficients and Molecular Friction Coefficients by Species ---')
            for sname, msd_s in msd_species.items():
                try:
                    # Convert MSD from A^2 to m^2
                    msd_s_m2 = msd_s * (1e-10)**2
                    D_s, slope_s, intercept_s = estimate_diffusion_from_msd(
                        times_species_msd, msd_s_m2, 
                        fit_start_frac=args.fit_start_frac, 
                        fit_end_frac=args.fit_end_frac
                    )
                    # Calculate friction coefficient using Stokes-Einstein relation: xi = kBT/D
                    if D_s > 0:
                        friction_coeff = (kB * args.T) / D_s  # Units: kg/s
                    else:
                        friction_coeff = float('inf')
                    
                    species_diffusion[sname] = {
                        'D_m2_s': D_s,
                        'slope': slope_s,
                        'intercept': intercept_s,
                        'friction_coeff_kg_s': friction_coeff
                    }
                    print(f'  {sname}: D = {D_s:.4e} m^2/s, xi = {friction_coeff:.4e} kg/s')
                except Exception as e:
                    print(f'  {sname}: Failed to fit diffusion coefficient: {e}')
                    species_diffusion[sname] = {
                        'D_m2_s': float('nan'),
                        'slope': float('nan'),
                        'intercept': float('nan'),
                        'friction_coeff_kg_s': float('nan')
                    }
            
            # Print molecular friction coefficient summary
            print('\n--- Molecular Friction Coefficient Analysis ---')
            valid_friction = {k: v for k, v in species_diffusion.items() 
                            if not np.isnan(v.get('friction_coeff_kg_s', float('nan')))}
            
            if valid_friction:
                friction_values = [v['friction_coeff_kg_s'] for v in valid_friction.values()]
                print(f'Number of species analyzed: {len(valid_friction)}')
                print(f'Molecular friction coefficient range: {min(friction_values):.2e} - {max(friction_values):.2e} kg/s')
                print(f'Average molecular friction coefficient: {np.mean(friction_values):.2e} kg/s')
                
                # Identify most and least mobile species
                min_friction_species = min(valid_friction.items(), key=lambda x: x[1]['friction_coeff_kg_s'])
                max_friction_species = max(valid_friction.items(), key=lambda x: x[1]['friction_coeff_kg_s'])
                
                print(f'Most mobile (lowest xi): {min_friction_species[0]} (xi = {min_friction_species[1]["friction_coeff_kg_s"]:.2e} kg/s)')
                print(f'Least mobile (highest xi): {max_friction_species[0]} (xi = {max_friction_species[1]["friction_coeff_kg_s"]:.2e} kg/s)')
            else:
                print('No valid molecular friction coefficients calculated.')
            
            # save species MSD to file with diffusion info in header
            prefix = args.out_prefix
            species_header = ['time_ps'] + [f'MSD_{s}_m2' for s in msd_species.keys()]
            msd_species_dat = f'{prefix}_msd_by_species.dat'
            
            # Create header with diffusion coefficients and friction coefficients
            header_lines = ['\t'.join(species_header)]
            for sname in msd_species.keys():
                if sname in species_diffusion:
                    D_val = species_diffusion[sname]['D_m2_s']
                    xi_val = species_diffusion[sname]['friction_coeff_kg_s']
                    header_lines.append(f'# {sname}: D = {D_val:.4e} m^2/s, xi = {xi_val:.4e} kg/s')
            header_text = '\n'.join(header_lines)
            
            # prepare columns: time_ps and each species converted to m^2
            arr_cols = [times_species_msd * 1e12] + [msd_species[s] * (1e-10)**2 for s in msd_species.keys()]
            out_arr = np.column_stack(arr_cols)
            np.savetxt(msd_species_dat, out_arr, header=header_text, comments='', delimiter='\t')
            print(T(f"[OK] 每种物种的 MSD 数据已保存到 {msd_species_dat}", f"[OK] Per-species MSD data saved to {msd_species_dat}"))
            
            # Save molecular friction coefficients to separate file
            molecular_friction_dat = f'{prefix}_molecular_friction.dat'
            molecular_friction_data = []
            molecular_friction_header = ['Species', 'Diffusion_Coeff_m2_s', 'Molecular_Friction_Coeff_kg_s', 'Molecular_Friction_Coeff_pN_s_m']
            
            for sname in msd_species.keys():
                if sname in species_diffusion:
                    D_val = species_diffusion[sname]['D_m2_s']
                    xi_val = species_diffusion[sname]['friction_coeff_kg_s']
                    # Convert to pN*s/m for easier reading (1 kg/s = 1e12 pN*s/m)
                    xi_pN = xi_val * 1e12 if not np.isnan(xi_val) else float('nan')
                    molecular_friction_data.append([sname, D_val, xi_val, xi_pN])
            
            # Save as DAT for Origin compatibility
            with open(molecular_friction_dat, 'w', encoding='utf-8') as f:
                f.write('\t'.join(molecular_friction_header) + '\n')
                for row in molecular_friction_data:
                    f.write(f'{row[0]}\t{row[1]:.6e}\t{row[2]:.6e}\t{row[3]:.6e}\n')
            print(T(f"[OK] 分子摩擦系数数据已保存到 {molecular_friction_dat}", f"[OK] Molecular-friction data saved to {molecular_friction_dat}"))
            
            # plot
            if not args.no_plots and HAS_MPL:
                save_plot_msd_by_species(times_species_msd, msd_species, f'{prefix}_msd_by_species.png', species_diffusion)
                save_plot_molecular_friction_coefficients(species_diffusion, f'{prefix}_molecular_friction.png')
        except Exception as e:
            print(f'Warning: Could not compute per-species MSD: {e}')
    else:
        if len(species) == 0:
            print('Info: Species information not available; skipping per-species MSD.')
        else:
            print('Warning: Species vector length mismatch; skipping per-species MSD.')

    # Stokes-Einstein viscosity
    eta_se = (kB * args.T) / (6.0 * math.pi * r_eff_m * D) if D > 0 else float('nan')
    print(f'Stokes-Einstein viscosity (using R={r_eff_A:.3f} A): eta_SE = {eta_se:.4e} Pa*s')

    # VACF (optional, use subset of frames for performance)
    print('\n--- Computing Velocity Autocorrelation Function (VACF) ---')
    vacf_stride = max(1, nframes // 2000)  # Use at most 2000 frames for VACF
    frac_coords_vacf = frac_coords[::vacf_stride]
    print(f'Using {frac_coords_vacf.shape[0]} frames (every {vacf_stride}th frame) for VACF calculation.')
    cart_coords_vacf = fractional_to_cart(unwrap_positions(frac_coords_vacf), lattice)
    dt_s_vacf = args.potim * vacf_stride * fs2s
    vacf_times, vacf = compute_vacf_from_positions(cart_coords_vacf, dt_s_vacf)
    print(f'VACF calculated for {len(vacf_times)} time lags.')
    
    # Compute Phonon Density of States (PDOS) from VACF
    print('\n--- Computing Phonon Density of States (PDOS) from VACF ---')
    try:
        frequencies_THz, pdos = compute_pdos_from_vacf(vacf_times, vacf, window='hann', max_freq_THz=100.0)
        print(f'PDOS calculated for {len(frequencies_THz)} frequency points (0 to {frequencies_THz[-1]:.2f} THz).')
        
        # Find peak frequency
        if len(pdos) > 0:
            peak_idx = np.argmax(pdos)
            peak_freq_THz = frequencies_THz[peak_idx]
            peak_freq_cm1 = peak_freq_THz * 33.356  # Convert THz to cm^-1
            print(f'Peak frequency: {peak_freq_THz:.3f} THz ({peak_freq_cm1:.1f} cm^-1)')
    except Exception as e:
        print(f'Warning: Could not compute PDOS from VACF: {e}')
        frequencies_THz = None
        pdos = None

    # Read stresses & energies
    print('\n--- Reading Stress and Energy Data ---')
    stress_arr = np.empty((0, 6))
    energy_list = []
    temps = []
    vol_series = None
    source = 'none'
    en_times = None
    if have_vasprun:
        try:
            print(f'Attempting to read vasprun.xml: {args.vasprun}')
            stress_arr_v, vol_v, energies_v = read_vasprun_stress_energy(args.vasprun)
            if stress_arr_v.size > 0:
                stress_arr = stress_arr_v
                source = 'vasprun'
            if vol_v is not None:
                vol_series = vol_v
            if len(energies_v) > 0:
                energy_list = energies_v
                if source == 'none':
                    source = 'vasprun'
            print(f'[OK] Parsed from {args.vasprun}: '
                  f'Stress: {stress_arr_v.shape[0]}, Energy: {len(energies_v)}, '
                  f'Volume: {len(vol_v) if vol_v is not None else 0}')
        except Exception as e:
            print(f'Error parsing vasprun.xml {args.vasprun}: {e}. Trying OUTCAR.')

    if stress_arr.size == 0 and have_outcar:
        try:
            print(f'Attempting to read OUTCAR: {args.outcar}')
            stress_arr_o, vol_o, energies_o, temps_o = read_outcar_stress_energy(args.outcar)
            if stress_arr_o.size > 0:
                stress_arr = stress_arr_o
            if vol_o is not None and vol_series is None:
                vol_series = vol_o
            if len(energies_o) > 0 and len(energy_list) == 0:
                energy_list = energies_o
            if len(temps_o) > 0 and len(temps) == 0:
                temps = temps_o
            if stress_arr_o.size > 0 or len(energies_o) > 0:
                if source == 'none':
                    source = 'outcar'
                print(f'[OK] Parsed from {args.outcar}: '
                      f'Stress: {stress_arr_o.shape[0]}, Energy: {len(energies_o)}, '
                      f'Temp: {len(temps_o)}, Volume: {len(vol_o) if vol_o is not None else 0}')
            else:
                print(f'Warning: No useful data found in {args.outcar}.')
            # Sanity check: only trim if BOTH stress and energy come from OUTCAR
            # Don't trim stress if energy is 0 — energy may come from OSZICAR later
            if stress_arr.size > 0 and len(energy_list) > 0:
                n_stress = stress_arr.shape[0]
                if len(energy_list) != n_stress:
                    n_min = min(len(energy_list), n_stress)
                    print(f'     Warning: Energy ({len(energy_list)}) != Stress ({n_stress}), trimming to {n_min}')
                    energy_list = energy_list[:n_min]
                    stress_arr = stress_arr[:n_min]
            if stress_arr.size > 0 and len(temps) > 0 and len(temps) != stress_arr.shape[0]:
                n_min = min(len(temps), stress_arr.shape[0])
                print(f'     Warning: Temp ({len(temps)}) != Stress ({stress_arr.shape[0]}), trimming to {n_min}')
                temps = temps[:n_min]
        except Exception as e:
            print(f'Error parsing OUTCAR {args.outcar}: {e}.')

    # If still no energy_list but OSZICAR present, try to parse it
    if (not energy_list or len(energy_list) == 0) and have_oszicar:
        try:
            print(f'Attempting to read OSZICAR: {args.oszicar}')
            energies_osz, times_osz, temps_osz = read_oszicar_energy(args.oszicar, args.potim)
            if len(energies_osz) > 0:
                energy_list = energies_osz
                en_times = times_osz
                if len(temps) == 0 and len(temps_osz) > 0:
                    temps = temps_osz #保存温度
                source = 'oszicar'
                print(f'[OK] Parsed {len(energies_osz)} energy entries from {args.oszicar}.')
            else:
                print(f'Warning: Could not parse energies from {args.oszicar}.')
        except Exception as e:
            print(f'Error parsing OSZICAR {args.oszicar}: {e}.')

    if stress_arr.size == 0:
        print('Warning: No stress series parsed from any source. Green-Kubo viscosity will be skipped.')

    # GK: prepare stress and compute running viscosity per block
    stable_start_idx = int(len(stress_arr) * args.stable_frac) if stress_arr.size else 0
    if stress_arr.size > 0:
        print(f'Discarding first {stable_start_idx} frames ({args.stable_frac*100:.1f}%) as transient for GK calculation.')

    if vol_series is not None:
        if len(vol_series) >= len(stress_arr):
            V_series_A3 = vol_series[:len(stress_arr)]
        else:
            V_series_A3 = np.concatenate([vol_series, np.full(len(stress_arr)-len(vol_series), vol_series[-1])])
        V_m3_series = V_series_A3 * A32m3
        V_mean_A3 = float(np.mean(V_series_A3))
        V_m3_use = float(np.mean(V_m3_series[stable_start_idx:]))
        print(f'Average volume used for GK calculation: {V_m3_use:.3e} m^3')
    else:
        V_mean_A3 = V_A3
        V_m3_use = V_m3
        print(f'Using initial cell volume for GK calculation: {V_m3_use:.3e} m^3 (no volume series found).')

    gk_results = None
    if stress_arr.size > 0:
        print('\n--- Computing Green-Kubo Viscosity ---')
        # use only post-stable frames
        stress_used = stress_arr[stable_start_idx:]
        nsteps_used = stress_used.shape[0]
        print(f'Using {nsteps_used} frames for Green-Kubo calculation after discarding transient.')
        # split into blocks
        bblocks = block_split_array(stress_used, args.n_blocks)
        visc_blocks = []
        visc_curves = []
        trunc_indices = []
        times_list = []
        sacf_list = []

        # dt_s for GK calculation should be based on the original POTIM, not strided
        dt_s_gk = args.potim * fs2s

        for bidx, bstress in enumerate(bblocks):
            if len(bstress) == 0:
                print(f'Warning: Block {bidx+1} is empty, skipping.')
                continue
            tvec, visc_run, ktr, sacf = compute_gk_viscosity_from_stress(
                bstress, V_m3_use, args.T, dt_s_gk,
                unbiased_acf=True, window=args.window, truncation=args.truncation, tmax_ps=args.tmax_ps)
            # use truncated value
            val = visc_run[min(ktr, len(visc_run)-1)] if len(visc_run)>0 else float('nan')
            visc_blocks.append(val)
            visc_curves.append(visc_run)
            trunc_indices.append(ktr)
            times_list.append(tvec)
            sacf_list.append(sacf)
            print(f'  Block {bidx+1}/{args.n_blocks}: Viscosity = {val:.4e} Pa*s (truncated at {tvec[ktr]*1e12:.2f} ps)')

        visc_blocks_arr = np.array(visc_blocks)
        visc_mean = float(np.mean(visc_blocks_arr)) if visc_blocks_arr.size>0 else float('nan')
        visc_std = float(np.std(visc_blocks_arr, ddof=1)) if visc_blocks_arr.size>1 else 0.0
        ci_low = ci_high = None
        if args.bootstrap and visc_blocks_arr.size>1:
            ci_low, ci_high = bootstrap_ci(visc_blocks_arr, ci=args.ci, n_bootstrap=args.n_bootstrap)
            print(f'Green-Kubo Viscosity (Bootstrap {args.ci*100:.0f}% CI): {visc_mean:.4e} Pa*s [{ci_low:.4e}, {ci_high:.4e}]')
        else:
            print(f'Green-Kubo Viscosity (Mean +/- Std): {visc_mean:.4e} +/- {visc_std:.4e} Pa*s')

        # mean curve (pad to min length)
        if len(visc_curves)>0:
            minlen = min(len(c) for c in visc_curves)
            mean_curve = np.mean([c[:minlen] for c in visc_curves], axis=0)
        else:
            mean_curve = np.array([])

        gk_results = {
            'blocks': visc_blocks_arr.tolist(),
            'mean': visc_mean,
            'std': visc_std,
            'ci': (ci_low, ci_high),
            'trunc_indices': trunc_indices,
            'times_list': times_list,
            'curves': visc_curves,
            'mean_curve': mean_curve,
            'sacf_list': sacf_list
        }

        # ---- Full-trajectory (non-blocked) GK analysis ----
        # Use the entire post-stable dataset as a single block to get
        # the complete time-range SACF and running viscosity curve.
        print(f'\n--- Full-trajectory Green-Kubo Analysis ({nsteps_used} frames, no block splitting) ---')
        full_times, full_visc_run, full_trunc_idx, full_sacf = compute_gk_viscosity_from_stress(
            stress_used, V_m3_use, args.T, dt_s_gk,
            unbiased_acf=True, window=args.window,
            truncation=args.truncation, tmax_ps=args.tmax_ps)
        full_visc_value = full_visc_run[min(full_trunc_idx, len(full_visc_run)-1)] if len(full_visc_run) > 0 else float('nan')
        full_duration_ps = full_times[-1] * 1e12 if len(full_times) > 0 else 0.0
        print(f'  Full-trajectory viscosity = {full_visc_value:.4e} Pa*s (duration: {full_duration_ps:.2f} ps)')
        print(f'  Block-averaged viscosity  = {visc_mean:.4e} Pa*s (for comparison)')

        gk_results['full_times'] = full_times
        gk_results['full_visc_running'] = full_visc_run
        gk_results['full_sacf'] = full_sacf
        gk_results['full_visc_value'] = full_visc_value
    
    # Compute hydrostatic pressure statistics from stress time series
    # (应力涨落弹性模量已移除: 仅凭应力时间序列无法得到真实弹性常数, 需 Born 项;
    #  弹性常数请用 SC-SS / SA-SS 应力-应变法。这里只保留物理意义明确的压力统计。)
    pressure_stats = None
    if stress_arr.size > 0:
        print('\n--- Computing Hydrostatic Pressure Statistics ---')
        pressure_stats = compute_pressure_statistics(stress_used)

        print(f'Mean Pressure:       {pressure_stats["pressure_mean_GPa"]:.4f} ± {pressure_stats["pressure_std_GPa"]:.4f} GPa')
        print('\nNote: 弹性模量请使用 SC-SS / SA-SS 应力-应变法; 应力涨落法缺 Born 项, 已不再输出 K/G/E/ν。')

    # -----------------------
    # Save results
    # -----------------------
    prefix = args.out_prefix
    os.makedirs('.', exist_ok=True)
    print(f'\n--- Saving Results (prefix: {prefix}) ---')

    # RDF data output
    rdf_dat = f'{prefix}_rdf.dat'
    rdf_header = f'r_A\tg_r\n# RDF Peak: r={peak_r:.3f} A, g(r)={peak_g:.3f}\n# First minimum: r={first_min_r:.3f} A\n# Coordination number (to first min): {coordination_number:.2f}\n# Coordination number (to peak): {cn_to_peak:.2f}\n# g(r) at large r: {g_r_large:.3f}'
    np.savetxt(rdf_dat, np.column_stack((r, g_r)), header=rdf_header, comments='', delimiter='\t')
    print(T(f"[OK] RDF数据已保存到 {rdf_dat}", f"[OK] RDF data saved to {rdf_dat}"))

    # MSD data output
    msd_dat = f'{prefix}_msd.dat'
    np.savetxt(msd_dat, np.column_stack((times_msd*1e12, msd_m2)), header='time_ps\tMSD_m2', comments='', delimiter='\t')
    print(T(f"[OK] MSD数据已保存到 {msd_dat}", f"[OK] MSD data saved to {msd_dat}"))

    # VACF data output
    vacf_dat = f'{prefix}_vacf.dat'
    np.savetxt(vacf_dat, np.column_stack((vacf_times*1e12, vacf)), header='lag_ps\tVACF_m2_s2', comments='', delimiter='\t')
    print(T(f"[OK] VACF数据已保存到 {vacf_dat}", f"[OK] VACF data saved to {vacf_dat}"))
    
    # PDOS data output
    if frequencies_THz is not None and pdos is not None:
        pdos_dat = f'{prefix}_pdos.dat'
        # Also save in cm^-1 for convenience
        frequencies_cm1 = frequencies_THz * 33.356
        pdos_header = 'freq_THz\tfreq_cm-1\tPDOS_normalized\n# Phonon Density of States from VACF Fourier Transform'
        np.savetxt(pdos_dat, np.column_stack((frequencies_THz, frequencies_cm1, pdos)), 
                  header=pdos_header, comments='', delimiter='\t')
        print(T(f"[OK] PDOS数据已保存到 {pdos_dat}", f"[OK] PDOS data saved to {pdos_dat}"))

    # Pressure statistics data output
    if pressure_stats is not None:
        pstat_dat = f'{prefix}_pressure_stats.dat'
        with open(pstat_dat, 'w', encoding='utf-8') as f:
            f.write('Property\tValue\tUnit\n')
            f.write(f'Mean_Pressure\t{pressure_stats["pressure_mean_GPa"]:.6f}\tGPa\n')
            f.write(f'Pressure_Std\t{pressure_stats["pressure_std_GPa"]:.6f}\tGPa\n')
        print(T(f"[OK] 压力统计数据已保存到 {pstat_dat}", f"[OK] Pressure-statistics data saved to {pstat_dat}"))

    # Save intermediate files by default (unless --no_intermediates is specified)
    save_intermediates = not args.no_intermediates
    if save_intermediates:
        # stress DAT for Origin compatibility
        stress_dat = f'{prefix}_stress.dat'
        if stress_arr.size > 0:
            header_stress = 'step\tPxx_Pa\tPyy_Pa\tPzz_Pa\tPxy_Pa\tPyz_Pa\tPzx_Pa'
            np.savetxt(stress_dat, np.column_stack((np.arange(len(stress_arr)), stress_arr)), header=header_stress, comments='', delimiter='\t')
            print(T(f"[OK] 应力数据已保存到 {stress_dat}", f"[OK] Stress data saved to {stress_dat}"))
        else:
            print(f"Info: No stress data to save to {stress_dat}")

        # acf and running viscosity for concatenated data
        if stress_arr.shape[0] > 0:
            # Use all 5 independent components (consistent with GK function)
            acf_xy = autocorr_fft(stress_used[:,3])
            acf_yz = autocorr_fft(stress_used[:,4])
            acf_xz = autocorr_fft(stress_used[:,5])
            Nxy_int = (stress_used[:,0] - stress_used[:,1]) / 2.0
            Nyz_int = (stress_used[:,1] - stress_used[:,2]) / 2.0
            acf_nxy = autocorr_fft(Nxy_int)
            acf_nyz = autocorr_fft(Nyz_int)
            acf_avg = (acf_xy + acf_yz + acf_xz + acf_nxy + acf_nyz) / 5.0
            lags = np.arange(len(acf_avg)) * dt_s_gk
            acf_dat = f'{prefix}_acf.dat'
            np.savetxt(acf_dat, np.column_stack((lags*1e12, acf_xy, acf_yz, acf_xz, acf_avg)),
                       header='lag_ps\tacf_xy\tacf_yz\tacf_xz\tacf_avg_5comp', comments='', delimiter='\t')
            print(T(f"[OK] ACF数据已保存到 {acf_dat}", f"[OK] ACF data saved to {acf_dat}"))

            visc_run_full = (V_m3_use / (kB * args.T)) * cumtrapz(acf_avg, lags, initial=0.0)
            visc_run_dat = f'{prefix}_visc_running.dat'
            np.savetxt(visc_run_dat, np.column_stack((lags*1e12, visc_run_full)), header='time_ps\tviscosity_Pa_s', comments='', delimiter='\t')
            print(T(f"[OK] Green-Kubo运行粘度数据已保存到 {visc_run_dat}", f"[OK] Green-Kubo running-viscosity data saved to {visc_run_dat}"))
        else:
            print("Info: No stress data available to save intermediate ACF and running viscosity.")

    # Save PNG plots
    if not args.no_plots:
        if HAS_MPL:
            save_plot_rdf(r, g_r, f'{prefix}_rdf.png')
            save_plot_msd(times_msd, msd_m2, D, f'{prefix}_msd.png')
            save_plot_vacf(vacf_times, vacf, f'{prefix}_vacf.png')
            
            # Plot PDOS if available
            if frequencies_THz is not None and pdos is not None:
                try:
                    save_plot_pdos(frequencies_THz, pdos, f'{prefix}_pdos.png')
                except Exception as e:
                    print(f"Warning: Error plotting PDOS: {e}")
            
            if msd_species is not None:
                # plot previously saved via save_plot_msd_by_species call in calculation step
                pass
            if gk_results is not None:
                times_list_local = [t for t in gk_results['times_list']]
                curves_local = [c for c in gk_results['curves']]
                save_plot_viscosity(times_list_local, curves_local, gk_results['mean_curve'], f'{prefix}_viscosity.png')
                
                # Save and plot SACF data
                if 'sacf_list' in gk_results and len(gk_results['sacf_list']) > 0:
                    # Calculate average SACF across all blocks
                    sacf_arrays = gk_results['sacf_list']
                    times_arrays = gk_results['times_list']
                    
                    # Find minimum length to ensure all arrays have same size
                    min_len = min(len(sacf) for sacf in sacf_arrays)
                    
                    # Average SACF across blocks
                    sacf_avg = np.mean([sacf[:min_len] for sacf in sacf_arrays], axis=0)
                    t_sacf = times_arrays[0][:min_len]  # Use time array from first block
                    
                    # Save averaged SACF data
                    np.savetxt(f"{prefix}_sacf.dat", 
                              np.column_stack((t_sacf*1e12, sacf_avg)),
                              header="time_ps\tSACF_avg", 
                              comments='', 
                              delimiter="\t")
                    
                    # Plot SACF
                    if HAS_MPL:
                        plt.figure(figsize=(6,4))
                        # Plot individual blocks with transparency
                        for i, sacf in enumerate(sacf_arrays):
                            t_block = times_arrays[i][:min_len]
                            plt.plot(t_block*1e12, sacf[:min_len], alpha=0.3, color='gray', 
                                   label='Individual blocks' if i == 0 else '')
                        # Plot average
                        plt.plot(t_sacf*1e12, sacf_avg, 'k-', linewidth=2, label='Average')
                        plt.xlabel("Time (ps)")
                        plt.ylabel("SACF (Pa^2)")
                        plt.title("Stress Autocorrelation Function (SACF)")
                        # Add text showing initial value
                        if len(sacf_avg) > 0:
                            plt.text(0.05, 0.95, f'SACF(0) = {sacf_avg[0]:.2e} Pa^2', 
                                   transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
                        plt.grid(True)
                        plt.legend()
                        plt.savefig(f"{prefix}_sacf.png", dpi=200)
                        plt.close()
                        print(f"SACF plot saved: {prefix}_sacf.png")
                    print(f"SACF data saved: {prefix}_sacf.dat")
                    
                    # Calculate integrated viscosity from SACF
                    # Note: autocorr_fft already normalizes SACF, so no additional normalization needed
                    # Get volume and temperature for integration
                    V_m3_for_integration = V_m3_use  # Use the same volume as in GK calculation
                    
                    # Integrate SACF to get viscosity: eta = (V / kBT) * integral(SACF(t) dt)
                    dt_integration = t_sacf[1] - t_sacf[0] if len(t_sacf) > 1 else args.potim * fs2s
                    integrated_viscosity = (V_m3_for_integration / (kB * args.T)) * cumtrapz(sacf_avg, t_sacf, initial=0.0)
                    
                    # Save integrated viscosity
                    np.savetxt(f"{prefix}_viscosity_from_sacf.dat", 
                              np.column_stack((t_sacf*1e12, integrated_viscosity)),
                              header="time_ps\tviscosity_from_SACF_Pa_s", 
                              comments='', 
                              delimiter="\t")
                    print(f"Integrated viscosity from SACF saved: {prefix}_viscosity_from_sacf.dat")
                    
                    # Plot integrated viscosity comparison
                    if HAS_MPL:
                        plt.figure(figsize=(8,6))
                        plt.plot(t_sacf*1e12, integrated_viscosity, 'r-', linewidth=2, 
                               label='Viscosity from SACF integration')
                        
                        # Add final GK viscosity as horizontal line for comparison
                        if gk_results and not np.isnan(gk_results['mean']):
                            plt.axhline(y=gk_results['mean'], color='b', linestyle='--', 
                                      label=f'GK viscosity (blocks): {gk_results["mean"]:.3e} Pa*s')
                        
                        plt.xlabel("Time (ps)")
                        plt.ylabel("Viscosity (Pa*s)")
                        plt.title("Viscosity from SACF Integration vs Block-averaged GK")
                        plt.grid(True)
                        plt.legend()
                        plt.savefig(f"{prefix}_viscosity_comparison.png", dpi=200)
                        plt.close()
                        print(f"Viscosity comparison plot saved: {prefix}_viscosity_comparison.png")
            else:
                print("Info: Green-Kubo results not available, skipping viscosity plot.")

            # ---- Save full-trajectory (non-blocked) GK data ----
            if gk_results is not None and 'full_times' in gk_results:
                ft = gk_results['full_times']
                fv = gk_results['full_visc_running']
                fs = gk_results['full_sacf']

                # Full SACF
                full_sacf_dat = f'{prefix}_full_sacf.dat'
                ft_lags = np.arange(len(fs)) * (args.potim * fs2s)
                np.savetxt(full_sacf_dat,
                           np.column_stack((ft_lags * 1e12, fs)),
                           header='time_ps\tSACF_full', comments='', delimiter='\t')
                print(T(f"[OK] 全程 SACF 数据已保存到 {full_sacf_dat} ({ft_lags[-1]*1e12:.2f} ps)", f"[OK] Full-trajectory SACF data saved to {full_sacf_dat} ({ft_lags[-1]*1e12:.2f} ps)"))

                # Full running viscosity
                full_visc_dat = f'{prefix}_full_visc_running.dat'
                np.savetxt(full_visc_dat,
                           np.column_stack((ft * 1e12, fv)),
                           header='time_ps\tviscosity_Pa_s', comments='', delimiter='\t')
                print(T(f"[OK] 全程 running viscosity 数据已保存到 {full_visc_dat} ({ft[-1]*1e12:.2f} ps)", f"[OK] Full-trajectory running-viscosity data saved to {full_visc_dat} ({ft[-1]*1e12:.2f} ps)"))

                # Full-trajectory plots
                if HAS_MPL:
                    # Full SACF plot
                    plt.figure(figsize=(10, 5))
                    plt.plot(ft_lags * 1e12, fs, 'b-', linewidth=1.5, label='Full-trajectory SACF')
                    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
                    plt.xlabel("Time (ps)")
                    plt.ylabel("SACF (Pa²)")
                    plt.title("Full-trajectory Stress Autocorrelation Function")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.savefig(f"{prefix}_full_sacf.png", dpi=200, bbox_inches='tight')
                    plt.close()
                    print(T(f"[OK] 全程 SACF 图已保存: {prefix}_full_sacf.png", f"[OK] Full-trajectory SACF figure saved: {prefix}_full_sacf.png"))

                    # Full running viscosity plot
                    plt.figure(figsize=(10, 5))
                    plt.plot(ft * 1e12, fv, 'r-', linewidth=1.5, label='Full-trajectory running viscosity')
                    if not np.isnan(gk_results['mean']):
                        plt.axhline(y=gk_results['mean'], color='b', linestyle='--',
                                    label=f'Block-averaged: {gk_results["mean"]:.3e} Pa·s')
                    plt.axhline(y=gk_results['full_visc_value'], color='g', linestyle=':',
                                label=f'Full-trajectory: {gk_results["full_visc_value"]:.3e} Pa·s')
                    plt.xlabel("Time (ps)")
                    plt.ylabel("Viscosity (Pa·s)")
                    plt.title("Full-trajectory Running Viscosity (Green-Kubo)")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.savefig(f"{prefix}_full_visc_running.png", dpi=200, bbox_inches='tight')
                    plt.close()
                    print(T(f"[OK] 全程 running viscosity 图已保存: {prefix}_full_visc_running.png", f"[OK] Full-trajectory running-viscosity figure saved: {prefix}_full_visc_running.png"))

            # energy vs time if available
            if len(energy_list) > 0:
                try:
                    if en_times is None:
                        # Construct time axis: each entry = one ionic step
                        en_times = np.arange(len(energy_list)) * args.potim * fs2s
                    # Ensure en_times matches energy_list length
                    n_en = min(len(en_times), len(energy_list))
                    en_times_use = en_times[:n_en]
                    energy_use = energy_list[:n_en]
                    temps_use = temps[:n_en] if len(temps) >= n_en else temps

                    save_plot_energy_time(en_times_use, energy_use, f'{prefix}_energy.png',
                                          temps=temps_use if len(temps_use) == n_en else None)
                    # Save energy data to file
                    energy_dat = f'{prefix}_energy.dat'
                    if len(temps_use) == n_en:
                        np.savetxt(energy_dat,
                                np.column_stack((en_times_use*1e12, temps_use, energy_use)),
                                header='time_ps\tT_K\tF_eV', comments='', delimiter='\t')
                        print(f"[OK] 能量-时间-温度数据已保存到 {energy_dat} ({n_en} 步)")
                    else:
                        np.savetxt(energy_dat,
                                np.column_stack((en_times_use*1e12, energy_use)),
                                header='time_ps\tF_eV', comments='', delimiter='\t')
                        print(f"[OK] 能量-时间数据已保存到 {energy_dat} ({n_en} 步)")
                except Exception as e:
                    print(f"Error saving energy plot/data: {e}")
            else:
                print("Info: No energy data available, skipping energy vs time plot and data save.")
            
            # 绘制SACF积分粘度图（如果数据文件存在）
            viscosity_from_sacf_file = f'{prefix}_viscosity_from_sacf.dat'
            if os.path.exists(viscosity_from_sacf_file):
                try:
                    save_plot_viscosity_from_sacf(viscosity_from_sacf_file, f'{prefix}_viscosity_from_sacf.png')
                except Exception as e:
                    print(f"Warning: Error plotting viscosity from SACF: {e}")

            # (应力涨落弹性模量图已移除; 压力统计只作为数值输出到 *_pressure_stats.dat)
        else:
            print("Warning: Matplotlib not available, skipping all plots.")
    else:
        print("Info: Plot generation skipped as requested by --no_plots flag.")

    # Save summary JSON
    summary = {
        'data_source': source,
        'T': args.T,
        'potim_fs': args.potim,
        'nframes': nframes,
        'natoms': natoms,
        'trajectory_used_steps_after_stable_window': stress_arr.shape[0] - stable_start_idx if stress_arr.size>0 else 0,
        'n_blocks': args.n_blocks,
        'rdf_peak_r_A': peak_r,
        'rdf_peak_g': peak_g,
        'rdf_first_min_r_A': first_min_r,
        'coordination_number_to_first_min': coordination_number,
        'coordination_number_to_peak': cn_to_peak,
        'rdf_g_at_large_r': g_r_large,
        'r_eff_A': r_eff_A,
        'diffusion_m2_s': D,
        'species_diffusion_coefficients': species_diffusion if species_diffusion else 'N/A',
        'stokes_einstein_viscosity_Pa_s': eta_se,
        'green_kubo_viscosity_Pa_s': gk_results['mean'] if gk_results else 'N/A',
        'green_kubo_viscosity_std_Pa_s': gk_results['std'] if gk_results else 'N/A',
        'green_kubo_viscosity_ci_low_Pa_s': gk_results['ci'][0] if gk_results and gk_results['ci'] else 'N/A',
        'green_kubo_viscosity_ci_high_Pa_s': gk_results['ci'][1] if gk_results and gk_results['ci'] else 'N/A',
        'green_kubo_full_trajectory_viscosity_Pa_s': gk_results['full_visc_value'] if gk_results and 'full_visc_value' in gk_results else 'N/A',
        'stress_source': source,
        'volume_A3': V_mean_A3,
        'density_g_cm3': density_g_cm3 if not np.isnan(density_g_cm3) else 'N/A',
        'energy_frames': len(energy_list),
        'temperature_frames': len(temps),
        'pressure_stats': pressure_stats if pressure_stats else 'N/A',
    }
    
    # Add friction calculation parameters to summary if provided
    if args.friction_U and args.friction_p and args.friction_h:
        summary['friction_velocities_m_s'] = [float(x) for x in args.friction_U.split()]
        summary['friction_pressure_Pa'] = args.friction_p
        summary['friction_film_thickness_m'] = args.friction_h

    json_name = f'{prefix}_summary.json'
    with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f'[OK] Summary JSON saved to {json_name}')
    
    # Generate Origin-friendly summary table
    save_summary_table(summary, prefix)
    
    # Advanced Analysis
    if (args.advanced_analysis or args.partial_rdf or args.structure_factor or 
        args.coordination_analysis or args.msd_moments or args.bond_analysis):
        
        print(f"\n{'='*60}")
        print(T("[ADVANCED] 开始高级分析...", "[ADVANCED] Starting advanced analysis..."))
        print(f"{'='*60}")
        
        # Initialize results containers
        partial_rdfs = {}
        structure_factor_result = None
        coordination_numbers = {}
        msd_moments_result = {}
        bond_lengths = {}
        bond_angles = {}
        
        # Check if we have species information
        if len(species) != natoms:
            print(T("[WARNING] 缺少物种信息，跳过需要物种信息的分析", "[WARNING] Missing species info; skipping species-dependent analysis"))
            print(T("   请确保POSCAR或XDATCAR包含元素符号信息", "   Please ensure POSCAR or XDATCAR contains element-symbol info"))
        else:
            print(T(f"[OK] 检测到 {len(set(species))} 种元素: {set(species)}", f"[OK] Detected {len(set(species))} element types: {set(species)}"))
            
            # Partial RDF analysis
            if args.advanced_analysis or args.partial_rdf:
                print(T("\n[ANALYSIS] 计算偏径向分布函数...", "\n[ANALYSIS] Computing partial radial distribution functions..."))
                try:
                    # Use subset of frames for efficiency
                    rdf_stride = max(1, nframes // 500)
                    frac_coords_analysis = frac_coords[::rdf_stride]
                    print(f"使用 {frac_coords_analysis.shape[0]} 帧进行偏RDF计算")
                    
                    partial_rdfs = compute_partial_rdf(frac_coords_analysis, lattice, species, 
                                                     rmax=min(8.0, np.max(np.linalg.norm(lattice, axis=1))/3.0))
                    print(f"[OK] 计算了 {len(partial_rdfs)} 个偏RDF")
                except Exception as e:
                    print(f"[ERROR] 偏RDF计算失败: {e}")
            
            # Structure factor analysis
            if args.advanced_analysis or args.structure_factor:
                print("\n[ANALYSIS] 计算静态结构因子...")
                try:
                    # Use subset of frames for efficiency
                    sf_stride = max(1, nframes // 200)
                    frac_coords_sf = frac_coords[::sf_stride]
                    print(f"使用 {frac_coords_sf.shape[0]} 帧进行结构因子计算")
                    
                    structure_factor_result = compute_structure_factor(frac_coords_sf, lattice, 
                                                                     qmax=args.qmax, nq=200)
                    print("[OK] 结构因子计算完成")
                except Exception as e:
                    print(f"[ERROR] 结构因子计算失败: {e}")
            
            # Coordination analysis
            if args.advanced_analysis or args.coordination_analysis:
                print(T("\n[ANALYSIS] 计算偏配位数...", "\n[ANALYSIS] Computing partial coordination numbers..."))
                try:
                    # Use subset of frames for efficiency
                    coord_stride = max(1, nframes // 100)
                    frac_coords_coord = frac_coords[::coord_stride]
                    print(T(f"使用 {frac_coords_coord.shape[0]} 帧进行配位数计算", f"Using {frac_coords_coord.shape[0]} frames for coordination-number calculation"))
                    
                    coordination_numbers = compute_partial_coordination_numbers(frac_coords_coord, lattice, species)
                    print(T(f"[OK] 计算了 {len(coordination_numbers)} 个偏配位数", f"[OK] Computed {len(coordination_numbers)} partial coordination numbers"))
                except Exception as e:
                    print(T(f"[ERROR] 配位数计算失败: {e}", f"[ERROR] Coordination-number calculation failed: {e}"))
            
            # Bond analysis
            if args.advanced_analysis or args.bond_analysis:
                print("\n[ANALYSIS] 计算键长和键角分布...")
                try:
                    # Use subset of frames for efficiency
                    bond_stride = max(1, nframes // 200)
                    frac_coords_bond = frac_coords[::bond_stride]
                    print(T(f"使用 {frac_coords_bond.shape[0]} 帧进行键分析", f"Using {frac_coords_bond.shape[0]} frames for bond analysis"))
                    
                    bond_lengths = compute_bond_length_distribution(frac_coords_bond, lattice, species, 
                                                                  r_max=args.bond_cutoff)
                    bond_angles = compute_bond_angle_distribution(frac_coords_bond, lattice, species, 
                                                                r_cutoff=args.bond_cutoff)
                    print(f"[OK] 键长分布: {len(bond_lengths)} 个对, 键角分布: {len(bond_angles)} 个中心原子")
                except Exception as e:
                    print(T(f"[ERROR] 键分析失败: {e}", f"[ERROR] Bond analysis failed: {e}"))
        
        # MSD moments analysis (doesn't require species info)
        if args.advanced_analysis or args.msd_moments:
            print(T("\n[ANALYSIS] 计算MSD高阶矩 (动力学非均匀性)...", "\n[ANALYSIS] Computing MSD higher moments (dynamic heterogeneity)..."))
            try:
                # Use subset of frames for efficiency
                moments_stride = max(1, nframes // 2000)
                frac_coords_moments = frac_coords[::moments_stride]
                print(T(f"使用 {frac_coords_moments.shape[0]} 帧进行MSD矩计算", f"Using {frac_coords_moments.shape[0]} frames for MSD-moment calculation"))
                
                msd_moments_result = compute_msd_higher_moments(frac_coords_moments, lattice, 
                                                              args.potim * moments_stride)
                print(T("[OK] MSD高阶矩计算完成", "[OK] MSD higher-moment calculation done"))
            except Exception as e:
                print(T(f"[ERROR] MSD高阶矩计算失败: {e}", f"[ERROR] MSD higher-moment calculation failed: {e}"))
        
        # Save results
        print(T(f"\n[SAVE] 保存高级分析结果...", f"\n[SAVE] Saving advanced analysis results..."))
        try:
            save_advanced_analysis_results(prefix, partial_rdfs, structure_factor_result,
                                         coordination_numbers, msd_moments_result,
                                         bond_lengths, bond_angles)
        except Exception as e:
            print(T(f"[ERROR] 保存高级分析结果失败: {e}", f"[ERROR] Failed to save advanced analysis results: {e}"))
        
        # Generate plots
        if not args.no_plots:
            print(T(f"\n[PLOT] 生成高级分析图表...", f"\n[PLOT] Generating advanced analysis plots..."))
            try:
                plot_advanced_analysis_results(prefix, partial_rdfs, structure_factor_result,
                                             msd_moments_result, bond_lengths, bond_angles)
            except Exception as e:
                print(T(f"[ERROR] 生成高级分析图表失败: {e}", f"[ERROR] Failed to generate advanced analysis plots: {e}"))
        
        print(T(f"\n[OK] 高级分析完成!", f"\n[OK] Advanced analysis done!"))
        print(f"{'='*60}")

    # Perform friction calculation if parameters are provided and viscosity is available
    final_viscosity = None
    if gk_results and not math.isnan(gk_results['mean']):
        final_viscosity = gk_results['mean']
        print(f'Using Green-Kubo viscosity for friction calculation: {final_viscosity:.4e} Pa*s')
    elif not math.isnan(eta_se):
        final_viscosity = eta_se
        print(f'Using Stokes-Einstein viscosity for friction calculation: {final_viscosity:.4e} Pa*s')
    else:
        print("Info: No valid viscosity (GK or SE) calculated. Cannot perform friction calculation.")

    if final_viscosity is not None and args.friction_U and args.friction_p and args.friction_h:
        print('\n--- Calculating Macroscopic Friction Coefficient (mu = eta*U/p*h) ---')
        try:
            U_list_friction = [float(x) for x in args.friction_U.split()]
            calc_friction(final_viscosity, U_list_friction, args.friction_p, args.friction_h, out_prefix=f'{prefix}_friction')
        except Exception as e:
            print(f"Error during friction calculation: {e}")
    elif args.friction_U or args.friction_p or args.friction_h:
        print("Warning: Friction calculation parameters provided, but no valid viscosity was calculated or some parameters are missing. Skipping friction calculation.")

    # Print comprehensive output summary
    print_output_summary(prefix)
    
    print("\n--- AIMD Post-processing Script Finished ---")


if __name__ == '__main__':
    main()

