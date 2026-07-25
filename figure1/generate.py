#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 1 generator for the GeoFriedmann 2D rock laminae paper.

The script reads all images from the folder "structures", computes
pore-space descriptors, builds Friedmann-inspired pore variables, and
generates a four-panel publication-style Figure 1.

Outputs:
    figure1_geofriedmann.png
    figure1_geofriedmann.pdf
    figure1_geofriedmann.svg
    figure1_descriptors.csv
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

OUTPUT_PNG = "figure1_geofriedmann.png"
OUTPUT_PDF = "figure1_geofriedmann.pdf"
OUTPUT_SVG = "figure1_geofriedmann.svg"
OUTPUT_CSV = "figure1_descriptors.csv"

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
# Utilities
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
    Reads an image and returns the solid mask.

    Default:
        green phase = solid/mineral phase
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


def percolation_from_labels(labels):
    """
    Checks horizontal and vertical percolation.
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
    Computes spatial anisotropy using PCA of occupied pixels.
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
    Computes basic metrics for one binary phase.
    """

    mask = mask.astype(bool)
    n_pixels = mask.size
    n_phase = int(mask.sum())

    labels, n_components = ndi.label(mask, structure=CONNECTIVITY_8)

    if n_components > 0:
        areas = np.bincount(labels.ravel())[1:].astype(float)

        area_mean = float(np.mean(areas))
        area_max = float(np.max(areas))
        largest_fraction = area_max / n_phase if n_phase > 0 else np.nan

    else:
        area_mean = np.nan
        area_max = 0.0
        largest_fraction = np.nan

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
        "area_max_px": area_max,
        "largest_component_fraction": largest_fraction,
        "perimeter_px": perimeter,
        "interface_density": interface_density,
        "euler_number": euler_number,
        "euler_density": euler_density,
        "percolates_x": percolates_x,
        "percolates_y": percolates_y,
        "anisotropy": anisotropy,
    }


def finite_difference(values, dt=1.0):
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
    values = np.asarray(values, dtype=float)
    output = np.full_like(values, np.nan)

    valid = np.isfinite(values) & (values > 0)
    output[valid] = np.log(values[valid])

    return output


def classify_regime(Kp, percolates, tolerance=0.5):
    if Kp > tolerance:
        return "Closed"

    if Kp < -tolerance:
        if percolates:
            return "Open"
        return "Transitional"

    return "Critical"


# ============================================================
# Descriptor computation
# ============================================================

def compute_descriptors(image_paths):
    rows = []
    solid_masks = []
    pore_masks = []

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
            "width_px": solid_mask.shape[1],
            "height_px": solid_mask.shape[0],

            "solid_fraction": solid["fraction"],
            "pore_fraction": pore["fraction"],

            "pore_area_mean_px": pore["area_mean_px"],
            "pore_largest_component_fraction": pore["largest_component_fraction"],
            "pore_interface_density": pore["interface_density"],
            "pore_euler_density": pore["euler_density"],
            "pore_percolates_x": pore["percolates_x"],
            "pore_percolates_y": pore["percolates_y"],
            "pore_anisotropy": pore["anisotropy"],

            "solid_area_mean_px": solid["area_mean_px"],
            "solid_largest_component_fraction": solid["largest_component_fraction"],
            "solid_interface_density": solid["interface_density"],
            "solid_euler_density": solid["euler_density"],
            "solid_percolates_x": solid["percolates_x"],
            "solid_percolates_y": solid["percolates_y"],
            "solid_anisotropy": solid["anisotropy"],
        }

        rows.append(row)
        solid_masks.append(solid_mask)
        pore_masks.append(pore_mask)

    df = pd.DataFrame(rows)

    # Pore scale factor
    pore_length = np.sqrt(df["pore_area_mean_px"].to_numpy(float))
    ap = np.full(len(df), np.nan)

    valid = np.isfinite(pore_length) & (pore_length > 0)

    if np.any(valid):
        ap[valid] = pore_length[valid] / pore_length[valid][0]

    Hp = finite_difference(safe_log(ap), dt=1.0)

    dap = finite_difference(ap, dt=1.0)
    d2ap = finite_difference(dap, dt=1.0)

    acceleration = np.full(len(df), np.nan)
    valid_acc = np.isfinite(d2ap) & np.isfinite(ap) & (ap > 0)
    acceleration[valid_acc] = d2ap[valid_acc] / ap[valid_acc]

    df["a_p"] = ap
    df["H_p"] = Hp
    df["acceleration_p"] = acceleration

    # Corrected pore curvature
    raw = df["pore_euler_density"].to_numpy(float)
    std = np.nanstd(raw)

    if std == 0 or not np.isfinite(std):
        z_euler = np.zeros_like(raw)
    else:
        z_euler = (raw - np.nanmean(raw)) / std

    percolation_penalty = (
        df["pore_percolates_x"].astype(int).to_numpy()
        + df["pore_percolates_y"].astype(int).to_numpy()
    )

    Kp = z_euler - percolation_penalty

    df["K_p"] = Kp
    df["pore_percolation_penalty"] = percolation_penalty

    # Reservoir potential index
    c0 = 0.25

    pore_percolates = (
        df["pore_percolates_x"].astype(bool)
        | df["pore_percolates_y"].astype(bool)
    )

    Cp = c0 + (1.0 - c0) * pore_percolates.astype(float)

    Rres = (
        df["pore_fraction"].to_numpy(float)
        * Cp
        * df["pore_largest_component_fraction"].to_numpy(float)
        * (1.0 + df["pore_anisotropy"].fillna(0).to_numpy(float))
    )

    df["pore_connectivity_factor"] = Cp
    df["R_res"] = Rres

    if np.nanmax(Rres) > np.nanmin(Rres):
        df["R_res_norm"] = (
            (Rres - np.nanmin(Rres))
            / (np.nanmax(Rres) - np.nanmin(Rres))
        )
    else:
        df["R_res_norm"] = np.ones_like(Rres)

    regimes = []

    for _, row in df.iterrows():
        percolates = bool(row["pore_percolates_x"]) or bool(row["pore_percolates_y"])
        regimes.append(classify_regime(row["K_p"], percolates))

    df["regime"] = regimes

    return df, solid_masks, pore_masks


# ============================================================
# Figure generation
# ============================================================

def add_panel_label(ax, label):
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left"
    )


def format_axis(ax):
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=2, length=6)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2)


def make_figure(df, pore_masks):
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    fig = plt.figure(figsize=FIGSIZE)

    gs = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.98,
        bottom=0.08,
        top=0.95,
        wspace=0.28,
        hspace=0.32
    )

    # ------------------------------------------------------------
    # Panel a: three rock laminae
    # ------------------------------------------------------------

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    add_panel_label(ax_a, "a")

    inset_positions = [
        [0.00, 0.12, 0.31, 0.76],
        [0.345, 0.12, 0.31, 0.76],
        [0.69, 0.12, 0.31, 0.76],
    ]

    n_show = min(3, len(pore_masks))

    for i in range(n_show):
        ax_in = ax_a.inset_axes(inset_positions[i])

        ax_in.imshow(pore_masks[i], cmap="gray", interpolation="nearest")
        ax_in.set_xticks([])
        ax_in.set_yticks([])

        for spine in ax_in.spines.values():
            spine.set_linewidth(2)

        ax_in.set_title(
            f"Stage {i + 1}",
            fontsize=LABEL_SIZE,
            fontweight="bold",
            pad=8
        )

    ax_a.text(
        0.5,
        0.01,
        "2D pore masks",
        ha="center",
        va="bottom",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        transform=ax_a.transAxes
    )

    # ------------------------------------------------------------
    # Panel b: porosity and connected pore fraction
    # ------------------------------------------------------------

    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "b")

    x = np.arange(len(df))
    width = 0.36

    ax_b.bar(
        x - width / 2,
        df["pore_fraction"],
        width,
        label="Porosity"
    )

    ax_b.bar(
        x + width / 2,
        df["pore_largest_component_fraction"],
        width,
        label="Largest pore network"
    )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{i}" for i in df["stage"]])

    ax_b.set_xlabel(
        "Stage",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_b.set_ylabel(
        "Fraction",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_b.set_title(
        "Pore Storage and Connectivity",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax_b.set_ylim(0, 1.08)
    ax_b.legend(frameon=False, fontsize=LEGEND_SIZE)
    ax_b.grid(axis="y", alpha=0.25)
    format_axis(ax_b)

    # ------------------------------------------------------------
    # Panel c: Friedmann phase space
    # ------------------------------------------------------------

    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "c")

    ax_c.plot(
        df["a_p"],
        df["H_p"],
        marker="o",
        linewidth=3,
        markersize=11
    )

    for _, row in df.iterrows():
        ax_c.text(
            row["a_p"],
            row["H_p"],
            f" {int(row['stage'])}",
            fontsize=LABEL_SIZE,
            fontweight="bold",
            va="center"
        )

    ax_c.set_xlabel(
        r"Pore scale factor $a_p$",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_c.set_ylabel(
        r"Evolution rate $H_p$",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_c.set_title(
        r"Friedmann Phase Space",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax_c.grid(alpha=0.25)
    format_axis(ax_c)

    # ------------------------------------------------------------
    # Panel d: curvature versus reservoir potential
    # ------------------------------------------------------------

    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "d")

    ax_d.axvline(0, linewidth=2, alpha=0.7)
    ax_d.axhline(0.5, linewidth=2, linestyle="--", alpha=0.7)

    sizes = 250 + 600 * df["pore_fraction"].to_numpy(float)

    ax_d.scatter(
        df["K_p"],
        df["R_res_norm"],
        s=sizes,
        edgecolor="black",
        linewidth=2
    )

    for _, row in df.iterrows():
        ax_d.text(
            row["K_p"],
            row["R_res_norm"],
            f" {int(row['stage'])}",
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
        r"Normalized reservoir index $R_{\mathrm{res}}$",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_d.set_title(
        "Regime and Reservoir Potential",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax_d.grid(alpha=0.25)
    format_axis(ax_d)

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

    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    image_paths = list_image_files(INPUT_DIR)

    print("Images used for Figure 1:")

    for path in image_paths:
        print(f"  - {path.name}")

    df, solid_masks, pore_masks = compute_descriptors(image_paths)

    df.to_csv(OUTPUT_CSV, index=False)

    make_figure(df, pore_masks)

    print()
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_SVG}")
    print(f"Saved: {OUTPUT_CSV}")

    print()
    print("Summary:")

    print(
        df[
            [
                "stage",
                "file",
                "pore_fraction",
                "pore_largest_component_fraction",
                "a_p",
                "H_p",
                "K_p",
                "R_res_norm",
                "regime"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
