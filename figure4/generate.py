#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 4 generator for the GeoFriedmann 2D rock laminae paper.

This figure performs a reservoir-quality screening of the three 2D
rock laminae using pore-space descriptors.

Panels:
    a) Storage-flow capacity map
    b) Normalized reservoir-quality component matrix
    c) Directional flow potential
    d) Final reservoir potential ranking

Inputs:
    structures/

Outputs:
    figure4_reservoir_screening.png
    figure4_reservoir_screening.pdf
    figure4_reservoir_screening.svg
    figure4_reservoir_screening.csv
"""

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure
from skimage.graph import MCP_Geometric


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = "structures"

OUTPUT_PNG = "figure4_reservoir_screening.png"
OUTPUT_PDF = "figure4_reservoir_screening.pdf"
OUTPUT_SVG = "figure4_reservoir_screening.svg"
OUTPUT_CSV = "figure4_reservoir_screening.csv"

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

MAX_TORTUOSITY_SIZE = 900


# ============================================================
# Image loading
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


def downsample_for_tortuosity(mask, max_size=900):
    """
    Downsamples a binary mask for faster shortest-path tortuosity.

    The operation uses simple striding. This keeps the code lightweight
    and avoids interpolation artifacts in binary masks.
    """

    height, width = mask.shape
    max_dim = max(height, width)

    if max_dim <= max_size:
        return mask.astype(bool), 1

    step = int(math.ceil(max_dim / max_size))
    mask_small = mask[::step, ::step].astype(bool)

    return mask_small, step


def tortuosity_mcp(mask, direction="x", max_size=900):
    """
    Approximates pore-network tortuosity using a shortest-path method.

    direction:
        x -> left-to-right flow
        y -> top-to-bottom flow

    tortuosity = shortest geodesic path / straight distance

    If no connected path exists, returns NaN.
    """

    mask = mask.astype(bool)
    mask, step = downsample_for_tortuosity(mask, max_size=max_size)

    height, width = mask.shape

    if height < 2 or width < 2:
        return np.nan

    if not mask.any():
        return np.nan

    if direction == "x":
        start_indices = np.flatnonzero(mask[:, 0])
        end_indices = np.flatnonzero(mask[:, -1])

        if len(start_indices) == 0 or len(end_indices) == 0:
            return np.nan

        starts = [(int(y), 0) for y in start_indices]
        ends = [(int(y), width - 1) for y in end_indices]

        straight_distance = width - 1

    elif direction == "y":
        start_indices = np.flatnonzero(mask[0, :])
        end_indices = np.flatnonzero(mask[-1, :])

        if len(start_indices) == 0 or len(end_indices) == 0:
            return np.nan

        starts = [(0, int(x)) for x in start_indices]
        ends = [(height - 1, int(x)) for x in end_indices]

        straight_distance = height - 1

    else:
        raise ValueError("direction must be either 'x' or 'y'.")

    if straight_distance <= 0:
        return np.nan

    cost = np.where(mask, 1.0, np.inf).astype(float)

    try:
        mcp = MCP_Geometric(cost, fully_connected=True)
        cumulative_costs, _ = mcp.find_costs(starts=starts, ends=ends)

        end_costs = np.array(
            [cumulative_costs[end] for end in ends],
            dtype=float
        )

        finite_costs = end_costs[np.isfinite(end_costs)]

        if len(finite_costs) == 0:
            return np.nan

        shortest_path = float(np.min(finite_costs))
        tortuosity = shortest_path / float(straight_distance)

        if tortuosity < 1.0:
            tortuosity = 1.0

        return tortuosity

    except Exception as exc:
        warnings.warn(f"Tortuosity computation failed: {exc}")
        return np.nan


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
# Reservoir descriptors
# ============================================================

def compute_reservoir_descriptors(image_paths):
    rows = []

    for stage, image_path in enumerate(image_paths, start=1):
        print(f"Processing Stage {stage}: {image_path.name}")

        solid_mask, detected_phase = read_binary_mask(image_path)
        pore_mask = ~solid_mask

        solid = phase_metrics(solid_mask)
        pore = phase_metrics(pore_mask)

        tau_x = np.nan
        tau_y = np.nan

        if pore["percolates_x"]:
            tau_x = tortuosity_mcp(
                pore_mask,
                direction="x",
                max_size=MAX_TORTUOSITY_SIZE
            )

        if pore["percolates_y"]:
            tau_y = tortuosity_mcp(
                pore_mask,
                direction="y",
                max_size=MAX_TORTUOSITY_SIZE
            )

        finite_tau = [
            value for value in [tau_x, tau_y]
            if np.isfinite(value) and value > 0
        ]

        if len(finite_tau) > 0:
            tau_eff = float(np.min(finite_tau))
        else:
            tau_eff = np.nan

        row = {
            "stage": stage,
            "file": image_path.name,
            "path": str(image_path),
            "detected_phase": detected_phase,

            "solid_fraction": solid["fraction"],
            "solid_largest_component_fraction": solid["largest_component_fraction"],
            "solid_interface_density": solid["interface_density"],
            "solid_euler_density": solid["euler_density"],

            "pore_fraction": pore["fraction"],
            "pore_largest_component_fraction": pore["largest_component_fraction"],
            "pore_area_mean_px": pore["area_mean_px"],
            "pore_interface_density": pore["interface_density"],
            "pore_euler_density": pore["euler_density"],
            "pore_percolates_x": pore["percolates_x"],
            "pore_percolates_y": pore["percolates_y"],
            "pore_anisotropy": pore["anisotropy"],

            "pore_tortuosity_x": tau_x,
            "pore_tortuosity_y": tau_y,
            "pore_tortuosity_effective": tau_eff,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

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
    # Reservoir potential index
    # ------------------------------------------------------------

    c0 = 0.25

    pore_percolates = (
        df["pore_percolates_x"].astype(bool)
        | df["pore_percolates_y"].astype(bool)
    )

    connectivity_factor = c0 + (1.0 - c0) * pore_percolates.astype(float)

    tau_for_score = df["pore_tortuosity_effective"].to_numpy(float)
    tau_for_score = np.where(
        np.isfinite(tau_for_score) & (tau_for_score > 0),
        tau_for_score,
        4.0
    )

    inverse_tortuosity = 1.0 / tau_for_score

    R_res = (
        df["pore_fraction"].to_numpy(float)
        * connectivity_factor
        * df["pore_largest_component_fraction"].to_numpy(float)
        * inverse_tortuosity
        * (1.0 + df["pore_anisotropy"].fillna(0).to_numpy(float))
    )

    df["pore_connectivity_factor"] = connectivity_factor
    df["inverse_tortuosity"] = inverse_tortuosity
    df["R_res"] = R_res
    df["R_res_norm"] = minmax(R_res)

    # Directional flow potential
    tau_x = df["pore_tortuosity_x"].to_numpy(float)
    tau_y = df["pore_tortuosity_y"].to_numpy(float)

    directional_x = np.where(
        np.isfinite(tau_x) & (tau_x > 0),
        df["pore_fraction"].to_numpy(float)
        * df["pore_largest_component_fraction"].to_numpy(float)
        / tau_x,
        0.0
    )

    directional_y = np.where(
        np.isfinite(tau_y) & (tau_y > 0),
        df["pore_fraction"].to_numpy(float)
        * df["pore_largest_component_fraction"].to_numpy(float)
        / tau_y,
        0.0
    )

    df["directional_flow_x"] = directional_x
    df["directional_flow_y"] = directional_y
    df["directional_flow_x_norm"] = minmax(directional_x)
    df["directional_flow_y_norm"] = minmax(directional_y)

    # Flow capacity used in scatter panel
    df["flow_capacity"] = (
        df["pore_largest_component_fraction"].to_numpy(float)
        * df["inverse_tortuosity"].to_numpy(float)
    )

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
    # Panel a: storage-flow map
    # ============================================================

    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "a")

    sizes = 300 + 900 * df["R_res_norm"].to_numpy(float)

    ax_a.scatter(
        df["pore_fraction"],
        df["flow_capacity"],
        s=sizes,
        edgecolor="black",
        linewidth=2
    )

    for _, row in df.iterrows():
        ax_a.text(
            row["pore_fraction"] + 0.01,
            row["flow_capacity"],
            str(int(row["stage"])),
            fontsize=LABEL_SIZE,
            fontweight="bold",
            va="center"
        )

    ax_a.set_xlabel(
        "Apparent porosity",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_a.set_ylabel(
        "Flow capacity proxy",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_a.set_title(
        "Storage-Flow Capacity Map",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax_a.grid(alpha=0.25)
    format_axis(ax_a)

    # ============================================================
    # Panel b: reservoir component matrix
    # ============================================================

    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "b")

    descriptor_names = [
        "Porosity",
        "Pore network",
        "Inverse tortuosity",
        "Open curvature",
        "Reservoir index",
    ]

    matrix = np.vstack([
        minmax(df["pore_fraction"].to_numpy(float)),
        minmax(df["pore_largest_component_fraction"].to_numpy(float)),
        minmax(df["inverse_tortuosity"].to_numpy(float)),
        minmax(-df["K_p"].to_numpy(float)),
        df["R_res_norm"].to_numpy(float),
    ])

    im = ax_b.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=1
    )

    ax_b.set_xticks(np.arange(len(df)))
    ax_b.set_xticklabels([str(s) for s in stages])

    ax_b.set_yticks(np.arange(len(descriptor_names)))
    ax_b.set_yticklabels(descriptor_names)

    ax_b.set_xlabel(
        "Stage",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_b.set_title(
        "Reservoir-Quality Components",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax_b.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold"
                )

    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized value", fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=13, width=2)

    format_axis(ax_b)

    # ============================================================
    # Panel c: directional flow potential
    # ============================================================

    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "c")

    x = np.arange(len(df))
    width = 0.36

    ax_c.bar(
        x - width / 2,
        df["directional_flow_x_norm"],
        width,
        label="Horizontal"
    )

    ax_c.bar(
        x + width / 2,
        df["directional_flow_y_norm"],
        width,
        label="Vertical"
    )

    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(s) for s in stages])

    ax_c.set_xlabel(
        "Stage",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_c.set_ylabel(
        "Normalized flow potential",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_c.set_title(
        "Directional Flow Potential",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax_c.set_ylim(0, 1.08)
    ax_c.legend(frameon=False, fontsize=LEGEND_SIZE)
    ax_c.grid(axis="y", alpha=0.25)
    format_axis(ax_c)

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

    ax_d.set_title(
        "Reservoir Potential Ranking",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

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

    print("Images used for Figure 4:")

    for path in image_paths:
        print(f"  - {path.name}")

    print()
    print("Computing reservoir descriptors...")
    print()

    df = compute_reservoir_descriptors(image_paths)

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
        "pore_tortuosity_x",
        "pore_tortuosity_y",
        "pore_tortuosity_effective",
        "K_p",
        "R_res_norm",
        "regime",
    ]

    print(df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
