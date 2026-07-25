#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3 generator for the GeoFriedmann 2D rock laminae paper.

This figure maps the three rock laminae into reduced-order geological
process affinities: pore opening / dissolution, cementation / occlusion,
and compaction / fragmentation.

Panels:
    a) GeoFriedmann process simplex
    b) Process affinity stacked bars
    c) Opening versus occlusion map
    d) Reservoir screening map

Inputs:
    structures/

Outputs:
    figure3_process_map.png
    figure3_process_map.pdf
    figure3_process_map.svg
    figure3_process_descriptors.csv
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = "structures"

OUTPUT_PNG = "figure3_process_map.png"
OUTPUT_PDF = "figure3_process_map.pdf"
OUTPUT_SVG = "figure3_process_map.svg"
OUTPUT_CSV = "figure3_process_descriptors.csv"

IMAGE_EXTENSIONS = [
    "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"
]

CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)

FIGSIZE = (16, 12)
DPI = 400

FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 22
LABEL_SIZE = 18
TICK_SIZE = 15
PANEL_LABEL_SIZE = 26
LEGEND_SIZE = 14


# ============================================================
# Image utilities
# ============================================================

def list_image_files(input_dir):
    input_dir = Path(input_dir)

    files = []

    for ext in IMAGE_EXTENSIONS:
        files.extend(input_dir.glob(ext))
        files.extend(input_dir.glob(ext.upper()))

    files = sorted(set(files), key=lambda p: str(p).lower())

    if len(files) == 0:
        raise FileNotFoundError(f"No images found in folder: {input_dir}")

    return files


def read_binary_mask(image_path):
    """
    Reads a 2D rock image and returns the solid mask.

    Convention:
        green phase = solid / mineral / facies phase
        black phase = pore phase
    """

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im).astype(np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green_mask = (
        (g > 40)
        & (g > r + 20)
        & (g > b + 20)
        & (g > 1.2 * r)
        & (g > 1.2 * b)
    )

    if green_mask.mean() > 0.001:
        solid_mask = green_mask.astype(bool)
        detected_phase = "green"
    else:
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

        if gray.max() == gray.min():
            threshold = gray.mean()
        else:
            threshold = filters.threshold_otsu(gray)

        solid_mask = gray > threshold
        detected_phase = "otsu_bright"

    return solid_mask, detected_phase


# ============================================================
# Descriptor functions
# ============================================================

def percolation_from_labels(labels):
    """
    Checks whether one connected component touches opposite boundaries.
    """

    if labels.max() == 0:
        return False, False

    left = np.unique(labels[:, 0])
    right = np.unique(labels[:, -1])

    top = np.unique(labels[0, :])
    bottom = np.unique(labels[-1, :])

    percolates_x = np.any(np.intersect1d(left, right) > 0)
    percolates_y = np.any(np.intersect1d(top, bottom) > 0)

    return bool(percolates_x), bool(percolates_y)


def anisotropy_pca(mask):
    """
    Computes spatial anisotropy using PCA of occupied pixel coordinates.
    """

    y, x = np.nonzero(mask)

    if len(x) < 2:
        return np.nan

    coords = np.column_stack([x, y]).astype(float)
    cov = np.cov(coords, rowvar=False)

    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]

    lambda_1 = eigvals[0]
    lambda_2 = eigvals[1]

    if lambda_1 + lambda_2 == 0:
        return np.nan

    return float((lambda_1 - lambda_2) / (lambda_1 + lambda_2))


def phase_metrics(mask):
    """
    Computes geometric, topological and connectivity descriptors.
    """

    mask = mask.astype(bool)

    n_pixels = mask.size
    n_phase = int(mask.sum())

    labels, n_components = ndi.label(mask, structure=CONNECTIVITY_8)

    if n_components > 0:
        areas = np.bincount(labels.ravel())[1:].astype(float)

        area_mean = float(np.mean(areas))
        area_median = float(np.median(areas))
        area_max = float(np.max(areas))

        largest_component_fraction = area_max / n_phase if n_phase > 0 else np.nan

    else:
        area_mean = np.nan
        area_median = np.nan
        area_max = 0.0
        largest_component_fraction = np.nan

    if n_phase > 0:
        perimeter = float(measure.perimeter_crofton(mask, directions=4))
        euler_number = int(measure.euler_number(mask, connectivity=2))
    else:
        perimeter = 0.0
        euler_number = 0

    interface_density = perimeter / n_pixels
    euler_density = euler_number / n_pixels

    percolates_x, percolates_y = percolation_from_labels(labels)
    anisotropy = anisotropy_pca(mask)

    return {
        "fraction": n_phase / n_pixels,
        "pixels": n_phase,
        "n_components": int(n_components),
        "area_mean_px": area_mean,
        "area_median_px": area_median,
        "area_max_px": area_max,
        "largest_component_fraction": largest_component_fraction,
        "perimeter_px": perimeter,
        "interface_density": interface_density,
        "euler_number": euler_number,
        "euler_density": euler_density,
        "percolates_x": percolates_x,
        "percolates_y": percolates_y,
        "anisotropy": anisotropy,
    }


def finite_difference(values, dt=1.0):
    """
    Numerical derivative using finite differences.
    """

    values = np.asarray(values, dtype=float)
    n = len(values)

    derivative = np.full(n, np.nan)

    if n < 2:
        return derivative

    for i in range(n):
        if i == 0:
            derivative[i] = (values[1] - values[0]) / dt
        elif i == n - 1:
            derivative[i] = (values[-1] - values[-2]) / dt
        else:
            derivative[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt)

    return derivative


def safe_log(values):
    """
    Natural logarithm only for finite positive values.
    """

    values = np.asarray(values, dtype=float)
    output = np.full_like(values, np.nan)

    valid = np.isfinite(values) & (values > 0)
    output[valid] = np.log(values[valid])

    return output


def minmax(values):
    """
    Min-max normalization.
    """

    values = np.asarray(values, dtype=float)
    output = np.full_like(values, np.nan)

    finite = np.isfinite(values)

    if not np.any(finite):
        return output

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if vmax == vmin:
        output[finite] = 1.0
    else:
        output[finite] = (values[finite] - vmin) / (vmax - vmin)

    return output


def classify_regime(Kp, percolates, tolerance=0.5):
    """
    Classifies pore-space regime from corrected curvature and percolation.
    """

    if Kp > tolerance:
        return "Closed"

    if Kp < -tolerance:
        if percolates:
            return "Open"
        return "Transitional"

    return "Critical"


# ============================================================
# GeoFriedmann process descriptors
# ============================================================

def compute_process_descriptors(image_paths):
    rows = []

    for stage, image_path in enumerate(image_paths, start=1):
        solid_mask, detected_phase = read_binary_mask(image_path)
        pore_mask = ~solid_mask

        solid = phase_metrics(solid_mask)
        pore = phase_metrics(pore_mask)

        row = {
            "stage": stage,
            "file": image_path.name,
            "path": str(image_path),
            "detected_phase": detected_phase,

            "solid_fraction": solid["fraction"],
            "solid_area_mean_px": solid["area_mean_px"],
            "solid_largest_component_fraction": solid["largest_component_fraction"],
            "solid_interface_density": solid["interface_density"],
            "solid_euler_density": solid["euler_density"],
            "solid_percolates_x": solid["percolates_x"],
            "solid_percolates_y": solid["percolates_y"],
            "solid_anisotropy": solid["anisotropy"],

            "pore_fraction": pore["fraction"],
            "pore_area_mean_px": pore["area_mean_px"],
            "pore_largest_component_fraction": pore["largest_component_fraction"],
            "pore_interface_density": pore["interface_density"],
            "pore_euler_density": pore["euler_density"],
            "pore_percolates_x": pore["percolates_x"],
            "pore_percolates_y": pore["percolates_y"],
            "pore_anisotropy": pore["anisotropy"],
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Pore scale factor and evolution rate
    # ------------------------------------------------------------

    pore_length = np.sqrt(df["pore_area_mean_px"].to_numpy(float))

    a_p = np.full(len(df), np.nan)

    valid = np.isfinite(pore_length) & (pore_length > 0)

    if np.any(valid):
        reference = pore_length[valid][0]
        a_p[valid] = pore_length[valid] / reference

    H_p = finite_difference(safe_log(a_p), dt=1.0)

    da_p = finite_difference(a_p, dt=1.0)
    d2a_p = finite_difference(da_p, dt=1.0)

    acceleration_p = np.full(len(df), np.nan)

    valid_acc = np.isfinite(d2a_p) & np.isfinite(a_p) & (a_p > 0)
    acceleration_p[valid_acc] = d2a_p[valid_acc] / a_p[valid_acc]

    df["a_p"] = a_p
    df["H_p"] = H_p
    df["acceleration_p"] = acceleration_p

    # ------------------------------------------------------------
    # Corrected pore curvature
    # ------------------------------------------------------------

    raw_euler = df["pore_euler_density"].to_numpy(float)
    std_euler = np.nanstd(raw_euler)

    if std_euler == 0 or not np.isfinite(std_euler):
        z_euler = np.zeros_like(raw_euler)
    else:
        z_euler = (raw_euler - np.nanmean(raw_euler)) / std_euler

    percolation_penalty = (
        df["pore_percolates_x"].astype(int).to_numpy()
        + df["pore_percolates_y"].astype(int).to_numpy()
    )

    K_p = z_euler - percolation_penalty

    df["pore_z_euler"] = z_euler
    df["pore_percolation_penalty"] = percolation_penalty
    df["K_p"] = K_p

    # ------------------------------------------------------------
    # Residual geological forcing
    # ------------------------------------------------------------

    phi = df["pore_fraction"].to_numpy(float)
    interface = df["pore_interface_density"].to_numpy(float)

    mean_phi = np.nanmean(phi)
    mean_interface = np.nanmean(interface)

    if np.isfinite(mean_interface) and mean_interface > 0:
        alpha = mean_phi / mean_interface
    else:
        alpha = 1.0

    Lambda_p = 3.0 * (
        df["acceleration_p"].to_numpy(float)
        + phi
        + alpha * interface
    )

    df["alpha_interface_scale"] = alpha
    df["Lambda_p"] = Lambda_p

    # ------------------------------------------------------------
    # Reservoir potential
    # ------------------------------------------------------------

    c0 = 0.25

    pore_percolates = (
        df["pore_percolates_x"].astype(bool)
        | df["pore_percolates_y"].astype(bool)
    )

    C_p = c0 + (1.0 - c0) * pore_percolates.astype(float)

    R_res = (
        df["pore_fraction"].to_numpy(float)
        * C_p
        * df["pore_largest_component_fraction"].to_numpy(float)
        * (1.0 + df["pore_anisotropy"].fillna(0).to_numpy(float))
    )

    df["pore_connectivity_factor"] = C_p
    df["R_res"] = R_res
    df["R_res_norm"] = minmax(R_res)

    # ------------------------------------------------------------
    # Process affinity scores
    # ------------------------------------------------------------

    porosity_n = minmax(df["pore_fraction"].to_numpy(float))
    largest_n = minmax(df["pore_largest_component_fraction"].to_numpy(float))
    interface_n = minmax(df["pore_interface_density"].to_numpy(float))
    anisotropy_n = minmax(df["pore_anisotropy"].fillna(0).to_numpy(float))
    open_curvature_n = minmax(-df["K_p"].to_numpy(float))
    closed_curvature_n = minmax(df["K_p"].to_numpy(float))
    inverse_porosity_n = minmax(1.0 - df["pore_fraction"].to_numpy(float))
    fragmentation_n = minmax(1.0 - df["pore_largest_component_fraction"].to_numpy(float))
    inverse_scale_n = minmax(1.0 / np.maximum(df["a_p"].to_numpy(float), 1e-12))

    opening_score = (
        0.30 * porosity_n
        + 0.25 * largest_n
        + 0.25 * open_curvature_n
        + 0.20 * df["R_res_norm"].to_numpy(float)
    )

    cementation_score = (
        0.35 * inverse_porosity_n
        + 0.25 * interface_n
        + 0.25 * closed_curvature_n
        + 0.15 * fragmentation_n
    )

    compaction_score = (
        0.35 * inverse_porosity_n
        + 0.30 * fragmentation_n
        + 0.20 * inverse_scale_n
        + 0.15 * closed_curvature_n
    )

    score_matrix = np.vstack([
        opening_score,
        cementation_score,
        compaction_score
    ]).T

    score_sum = np.sum(score_matrix, axis=1)
    score_sum[score_sum == 0] = 1.0

    normalized_scores = score_matrix / score_sum[:, None]

    df["opening_affinity"] = normalized_scores[:, 0]
    df["cementation_affinity"] = normalized_scores[:, 1]
    df["compaction_affinity"] = normalized_scores[:, 2]

    dominant_process = []

    for _, row in df.iterrows():
        values = {
            "Opening": row["opening_affinity"],
            "Cementation": row["cementation_affinity"],
            "Compaction": row["compaction_affinity"],
        }

        dominant_process.append(max(values, key=values.get))

    df["dominant_process"] = dominant_process

    regimes = []

    for _, row in df.iterrows():
        percolates = bool(row["pore_percolates_x"]) or bool(row["pore_percolates_y"])
        regimes.append(classify_regime(row["K_p"], percolates))

    df["regime"] = regimes

    return df


# ============================================================
# Figure utilities
# ============================================================

def add_panel_label(ax, label):
    ax.text(
        -0.10,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="top"
    )


def format_axis(ax):
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=2, length=6)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


# ============================================================
# Figure generation
# ============================================================

def make_figure(df):
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    fig = plt.figure(figsize=FIGSIZE)

    gs = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.95,
        wspace=0.34,
        hspace=0.34
    )

    stages = df["stage"].to_numpy(int)

    # ============================================================
    # Panel a: process simplex
    # ============================================================

    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "a")

    opening_vertex = np.array([0.00, 0.00])
    cementation_vertex = np.array([1.00, 0.00])
    compaction_vertex = np.array([0.50, math.sqrt(3) / 2.0])

    triangle = np.vstack([
        opening_vertex,
        cementation_vertex,
        compaction_vertex,
        opening_vertex
    ])

    ax_a.plot(
        triangle[:, 0],
        triangle[:, 1],
        linewidth=3
    )

    barycentric = []

    for _, row in df.iterrows():
        point = (
            row["opening_affinity"] * opening_vertex
            + row["cementation_affinity"] * cementation_vertex
            + row["compaction_affinity"] * compaction_vertex
        )

        barycentric.append(point)

    barycentric = np.asarray(barycentric)

    sizes = 280 + 700 * df["R_res_norm"].to_numpy(float)

    ax_a.scatter(
        barycentric[:, 0],
        barycentric[:, 1],
        s=sizes,
        edgecolor="black",
        linewidth=2,
        zorder=3
    )

    for i, row in df.iterrows():
        ax_a.text(
            barycentric[i, 0] + 0.025,
            barycentric[i, 1] + 0.020,
            str(int(row["stage"])),
            fontsize=LABEL_SIZE,
            fontweight="bold"
        )

    ax_a.text(
        opening_vertex[0] - 0.02,
        opening_vertex[1] - 0.07,
        "Opening",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        ha="center"
    )

    ax_a.text(
        cementation_vertex[0] + 0.02,
        cementation_vertex[1] - 0.07,
        "Cementation",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        ha="center"
    )

    ax_a.text(
        compaction_vertex[0],
        compaction_vertex[1] + 0.06,
        "Compaction",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        ha="center"
    )

    ax_a.set_title(
        "GeoFriedmann Process Simplex",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax_a.set_xlim(-0.12, 1.12)
    ax_a.set_ylim(-0.12, 1.02)
    ax_a.set_aspect("equal")
    ax_a.axis("off")

    # ============================================================
    # Panel b: process affinity bars
    # ============================================================

    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "b")

    x = np.arange(len(df))

    opening = df["opening_affinity"].to_numpy(float)
    cementation = df["cementation_affinity"].to_numpy(float)
    compaction = df["compaction_affinity"].to_numpy(float)

    ax_b.bar(
        x,
        opening,
        label="Opening"
    )

    ax_b.bar(
        x,
        cementation,
        bottom=opening,
        label="Cementation"
    )

    ax_b.bar(
        x,
        compaction,
        bottom=opening + cementation,
        label="Compaction"
    )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([str(s) for s in stages])

    ax_b.set_xlabel(
        "Stage",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_b.set_ylabel(
        "Process affinity",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_b.set_title(
        "Reduced-Order Process Attribution",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax_b.set_ylim(0, 1.05)
    ax_b.legend(frameon=False, fontsize=LEGEND_SIZE, loc="upper right")
    ax_b.grid(axis="y", alpha=0.25)
    format_axis(ax_b)

    # ============================================================
    # Panel c: opening versus occlusion
    # ============================================================

    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "c")

    opening_axis = df["opening_affinity"].to_numpy(float)
    occlusion_axis = (
        df["cementation_affinity"].to_numpy(float)
        + df["compaction_affinity"].to_numpy(float)
    )

    ax_c.plot(
        [0, 1],
        [1, 0],
        linestyle="--",
        linewidth=2,
        alpha=0.8
    )

    ax_c.scatter(
        opening_axis,
        occlusion_axis,
        s=280 + 700 * df["R_res_norm"].to_numpy(float),
        edgecolor="black",
        linewidth=2
    )

    for _, row in df.iterrows():
        ax_c.text(
            row["opening_affinity"] + 0.025,
            row["cementation_affinity"] + row["compaction_affinity"],
            str(int(row["stage"])),
            fontsize=LABEL_SIZE,
            fontweight="bold",
            va="center"
        )

    ax_c.set_xlabel(
        "Pore-opening affinity",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_c.set_ylabel(
        "Occlusion affinity",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_c.set_title(
        "Opening versus Occlusion",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax_c.set_xlim(0, 1.05)
    ax_c.set_ylim(0, 1.05)
    ax_c.grid(alpha=0.25)
    format_axis(ax_c)

    # ============================================================
    # Panel d: reservoir screening map
    # ============================================================

    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "d")

    ax_d.axvline(
        0,
        linewidth=2,
        alpha=0.7
    )

    ax_d.axhline(
        0.5,
        linewidth=2,
        linestyle="--",
        alpha=0.8
    )

    sizes = 280 + 700 * df["pore_fraction"].to_numpy(float)

    ax_d.scatter(
        df["K_p"],
        df["R_res_norm"],
        s=sizes,
        edgecolor="black",
        linewidth=2
    )

    for _, row in df.iterrows():
        ax_d.text(
            row["K_p"] + 0.04,
            row["R_res_norm"],
            str(int(row["stage"])),
            fontsize=LABEL_SIZE,
            fontweight="bold",
            va="center"
        )

    ax_d.set_xlabel(
        r"Corrected pore curvature $K_p$",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_d.set_ylabel(
        r"Normalized reservoir potential $R_{\mathrm{res}}$",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_d.set_title(
        "Reservoir Screening Map",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax_d.text(
        0.04,
        0.94,
        "Closed",
        transform=ax_d.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top"
    )

    ax_d.text(
        0.96,
        0.94,
        "Open",
        transform=ax_d.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="right",
        va="top"
    )

    ax_d.text(
        0.96,
        0.08,
        "Low potential",
        transform=ax_d.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="right",
        va="bottom"
    )

    ax_d.grid(alpha=0.25)
    format_axis(ax_d)

    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    image_paths = list_image_files(INPUT_DIR)

    print("Images used for Figure 3:")

    for path in image_paths:
        print(f"  - {path.name}")

    df = compute_process_descriptors(image_paths)

    df.to_csv(OUTPUT_CSV, index=False)

    make_figure(df)

    print()
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_SVG}")
    print(f"Saved: {OUTPUT_CSV}")

    print()
    print("Summary:")

    summary_cols = [
        "stage",
        "file",
        "pore_fraction",
        "pore_largest_component_fraction",
        "a_p",
        "H_p",
        "K_p",
        "Lambda_p",
        "R_res_norm",
        "opening_affinity",
        "cementation_affinity",
        "compaction_affinity",
        "dominant_process",
        "regime",
    ]

    print(df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
