#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 2 generator for the GeoFriedmann 2D rock laminae paper.

This figure summarizes the reduced-order GeoFriedmann diagnostic model.

Panels:
    a) Normalized descriptor matrix
    b) Pore scale factor and evolution rate
    c) Corrected pore curvature and residual forcing
    d) Reservoir potential ranking

Inputs:
    structures/

Outputs:
    figure2_geofriedmann_diagnostics.png
    figure2_geofriedmann_diagnostics.pdf
    figure2_geofriedmann_diagnostics.svg
    figure2_diagnostics.csv
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

OUTPUT_PNG = "figure2_geofriedmann_diagnostics.png"
OUTPUT_PDF = "figure2_geofriedmann_diagnostics.pdf"
OUTPUT_SVG = "figure2_geofriedmann_diagnostics.svg"
OUTPUT_CSV = "figure2_diagnostics.csv"

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

    Default convention:
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
    Checks whether a connected component touches opposite boundaries.
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

    anisotropy = (lambda_1 - lambda_2) / (lambda_1 + lambda_2)

    return float(anisotropy)


def phase_metrics(mask):
    """
    Computes basic geometric, topological and connectivity descriptors.
    """

    mask = mask.astype(bool)
    n_pixels = mask.size
    n_phase = int(mask.sum())

    labels, n_components = ndi.label(mask, structure=CONNECTIVITY_8)

    if n_components > 0:
        areas = np.bincount(labels.ravel())[1:].astype(float)

        area_mean = float(np.mean(areas))
        area_max = float(np.max(areas))
        area_median = float(np.median(areas))

        largest_component_fraction = area_max / n_phase if n_phase > 0 else np.nan

    else:
        area_mean = np.nan
        area_max = 0.0
        area_median = np.nan
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
    Classifies the pore-space regime from corrected curvature and percolation.
    """

    if Kp > tolerance:
        return "Closed"

    if Kp < -tolerance:
        if percolates:
            return "Open"
        return "Transitional"

    return "Critical"


# ============================================================
# GeoFriedmann diagnostics
# ============================================================

def compute_diagnostics(image_paths):
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
            "solid_largest_component_fraction": solid["largest_component_fraction"],
            "solid_area_mean_px": solid["area_mean_px"],
            "solid_interface_density": solid["interface_density"],
            "solid_euler_density": solid["euler_density"],
            "solid_percolates_x": solid["percolates_x"],
            "solid_percolates_y": solid["percolates_y"],
            "solid_anisotropy": solid["anisotropy"],

            "pore_fraction": pore["fraction"],
            "pore_largest_component_fraction": pore["largest_component_fraction"],
            "pore_area_mean_px": pore["area_mean_px"],
            "pore_interface_density": pore["interface_density"],
            "pore_euler_density": pore["euler_density"],
            "pore_percolates_x": pore["percolates_x"],
            "pore_percolates_y": pore["percolates_y"],
            "pore_anisotropy": pore["anisotropy"],
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Pore scale factor
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
    # Residual geological forcing Lambda_p
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
    # Reservoir potential index
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


def format_twin_axis(ax):
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
        wspace=0.35,
        hspace=0.34
    )

    stages = df["stage"].to_numpy(int)
    stage_labels = [str(s) for s in stages]

    # ============================================================
    # Panel a: descriptor matrix
    # ============================================================

    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "a")

    descriptor_names = [
        "Porosity",
        "Largest pore network",
        "Interface density",
        "Pore anisotropy",
        "Open curvature",
        "Reservoir index",
    ]

    values = np.vstack([
        minmax(df["pore_fraction"].to_numpy(float)),
        minmax(df["pore_largest_component_fraction"].to_numpy(float)),
        minmax(df["pore_interface_density"].to_numpy(float)),
        minmax(df["pore_anisotropy"].fillna(0).to_numpy(float)),
        minmax(-df["K_p"].to_numpy(float)),
        df["R_res_norm"].to_numpy(float),
    ])

    im = ax_a.imshow(values, aspect="auto", vmin=0, vmax=1)

    ax_a.set_xticks(np.arange(len(stages)))
    ax_a.set_xticklabels(stage_labels)

    ax_a.set_yticks(np.arange(len(descriptor_names)))
    ax_a.set_yticklabels(descriptor_names)

    ax_a.set_xlabel("Stage", fontsize=LABEL_SIZE, fontweight="bold")
    ax_a.set_title("Normalized Diagnostic Matrix", fontsize=TITLE_SIZE, fontweight="bold")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_value = values[i, j]
            if np.isfinite(text_value):
                ax_a.text(
                    j,
                    i,
                    f"{text_value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold"
                )

    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13, width=2)
    cbar.set_label("Normalized value", fontsize=14, fontweight="bold")

    format_axis(ax_a)

    # ============================================================
    # Panel b: scale factor and evolution rate
    # ============================================================

    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "b")

    ax_b.plot(
        stages,
        df["a_p"],
        marker="o",
        linewidth=3,
        markersize=11,
        label=r"$a_p$"
    )

    ax_b.set_xlabel("Stage", fontsize=LABEL_SIZE, fontweight="bold")
    ax_b.set_ylabel(r"Pore scale factor $a_p$", fontsize=LABEL_SIZE, fontweight="bold")
    ax_b.set_title("Pore-Space Expansion", fontsize=TITLE_SIZE, fontweight="bold")
    ax_b.set_xticks(stages)
    ax_b.grid(alpha=0.25)
    format_axis(ax_b)

    ax_b2 = ax_b.twinx()

    ax_b2.plot(
        stages,
        df["H_p"],
        marker="s",
        linewidth=3,
        markersize=10,
        linestyle="--",
        label=r"$H_p$"
    )

    ax_b2.set_ylabel(r"Evolution rate $H_p$", fontsize=LABEL_SIZE, fontweight="bold")
    format_twin_axis(ax_b2)

    lines_1, labels_1 = ax_b.get_legend_handles_labels()
    lines_2, labels_2 = ax_b2.get_legend_handles_labels()

    ax_b.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="best"
    )

    # ============================================================
    # Panel c: curvature and residual forcing
    # ============================================================

    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "c")

    ax_c.axhline(0, linewidth=2, alpha=0.7)

    ax_c.bar(
        stages,
        df["K_p"],
        width=0.55,
        label=r"$K_p$"
    )

    ax_c.set_xlabel("Stage", fontsize=LABEL_SIZE, fontweight="bold")
    ax_c.set_ylabel(r"Corrected curvature $K_p$", fontsize=LABEL_SIZE, fontweight="bold")
    ax_c.set_title("Topology and Residual Forcing", fontsize=TITLE_SIZE, fontweight="bold")
    ax_c.set_xticks(stages)
    ax_c.grid(axis="y", alpha=0.25)
    format_axis(ax_c)

    ax_c2 = ax_c.twinx()

    ax_c2.plot(
        stages,
        df["Lambda_p"],
        marker="D",
        linewidth=3,
        markersize=9,
        linestyle="--",
        label=r"$\Lambda_p$"
    )

    ax_c2.set_ylabel(r"Residual forcing $\Lambda_p$", fontsize=LABEL_SIZE, fontweight="bold")
    format_twin_axis(ax_c2)

    lines_1, labels_1 = ax_c.get_legend_handles_labels()
    lines_2, labels_2 = ax_c2.get_legend_handles_labels()

    ax_c.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="best"
    )

    # ============================================================
    # Panel d: reservoir ranking
    # ============================================================

    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "d")

    df_rank = df.sort_values("R_res_norm", ascending=True).reset_index(drop=True)

    y = np.arange(len(df_rank))

    ax_d.barh(
        y,
        df_rank["R_res_norm"],
        height=0.55
    )

    ax_d.set_yticks(y)
    ax_d.set_yticklabels([f"Stage {int(s)}" for s in df_rank["stage"]])

    ax_d.set_xlabel(
        r"Normalized reservoir potential $R_{\mathrm{res}}$",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_d.set_title("Reservoir Potential Ranking", fontsize=TITLE_SIZE, fontweight="bold")
    ax_d.set_xlim(0, 1.08)
    ax_d.grid(axis="x", alpha=0.25)

    for i, row in df_rank.iterrows():
        ax_d.text(
            row["R_res_norm"] + 0.03,
            i,
            row["regime"],
            va="center",
            ha="left",
            fontsize=14,
            fontweight="bold"
        )

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

    print("Images used for Figure 2:")

    for path in image_paths:
        print(f"  - {path.name}")

    df = compute_diagnostics(image_paths)

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
        "regime",
    ]

    print(df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
