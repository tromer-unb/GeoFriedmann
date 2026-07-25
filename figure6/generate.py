#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 6 generator for the GeoFriedmann 2D rock laminae paper.

This figure creates a synthetic temporal evolution experiment from one
representative 2D rock lamina. The first selected rock image is treated
as t = 0 years, and two idealized geological scenarios are generated:

1. Pore opening / dissolution
2. Cementation / pore occlusion

For each scenario, synthetic states are generated at:
    0, 100, 200, 300 years

The GeoFriedmann descriptor system is then applied to the synthetic
time sequence to show how the framework can track structural evolution,
reservoir tendency, and topological regime shifts.

Panels:
    a) Synthetic pore opening sequence
    b) GeoFriedmann diagnostics for pore opening
    c) Synthetic cementation sequence
    d) GeoFriedmann diagnostics for cementation

Inputs:
    structures/

Outputs:
    figure6_synthetic_time_evolution.png
    figure6_synthetic_time_evolution.pdf
    figure6_synthetic_time_evolution.svg
    figure6_synthetic_time_evolution.csv
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

OUTPUT_PNG = "figure6_synthetic_time_evolution.png"
OUTPUT_PDF = "figure6_synthetic_time_evolution.pdf"
OUTPUT_SVG = "figure6_synthetic_time_evolution.svg"
OUTPUT_CSV = "figure6_synthetic_time_evolution.csv"

IMAGE_EXTENSIONS = [
    "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"
]

CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)

FIGSIZE = (18, 12)
DPI = 400

FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 22
LABEL_SIZE = 18
TICK_SIZE = 15
PANEL_LABEL_SIZE = 26
LEGEND_SIZE = 13

BASE_IMAGE_INDEX = 0
TIME_YEARS = [0, 100, 200, 300]

BASELINE_CONNECTIVITY = 0.25
NON_PERCOLATING_TORTUOSITY = 4.0
MAX_TORTUOSITY_SIZE = 700


# ============================================================
# File handling
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


def read_solid_mask(image_path):
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

    return solid_mask.astype(bool), detected_phase


# ============================================================
# Synthetic evolution
# ============================================================

def default_step_pixels(mask):
    """
    Chooses a reasonable perturbation step from image size.

    For a 4096 px image, this returns 4 px.
    """

    min_dim = min(mask.shape)
    step = max(1, int(round(min_dim / 1024.0)))
    return step


def perturb_solid_mask(solid_mask, scenario, iterations):
    """
    Generates synthetic evolution states.

    scenario = "opening"
        erodes the solid phase, representing pore opening / dissolution.

    scenario = "cementation"
        dilates the solid phase, representing mineral growth / pore closure.
    """

    solid_mask = solid_mask.astype(bool)

    if iterations == 0:
        return solid_mask.copy()

    structure = np.ones((3, 3), dtype=bool)

    if scenario == "opening":
        out = ndi.binary_erosion(
            solid_mask,
            structure=structure,
            iterations=int(iterations)
        )
    elif scenario == "cementation":
        out = ndi.binary_dilation(
            solid_mask,
            structure=structure,
            iterations=int(iterations)
        )
    else:
        raise ValueError("scenario must be 'opening' or 'cementation'.")

    return out.astype(bool)


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


def downsample_for_tortuosity(mask, max_size=700):
    """
    Downsamples a binary mask for faster shortest-path tortuosity.
    """

    height, width = mask.shape
    max_dim = max(height, width)

    if max_dim <= max_size:
        return mask.astype(bool)

    factor = int(math.ceil(max_dim / max_size))
    return mask[::factor, ::factor].astype(bool)


def tortuosity_mcp(mask, direction="x", max_size=700):
    """
    Approximates pore-network tortuosity using a shortest-path method.

    direction:
        x -> left-to-right flow
        y -> top-to-bottom flow

    tortuosity = shortest geodesic path / straight distance
    """

    mask = mask.astype(bool)
    mask = downsample_for_tortuosity(mask, max_size=max_size)

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

        end_costs = np.array([cumulative_costs[end] for end in ends], dtype=float)
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


def safe_log(values):
    """
    Natural logarithm only for finite positive values.
    """

    values = np.asarray(values, dtype=float)

    output = np.full_like(values, np.nan)
    valid = np.isfinite(values) & (values > 0)

    output[valid] = np.log(values[valid])

    return output


def finite_difference(values, dt):
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
# Series computation
# ============================================================

def compute_series(base_solid_mask, base_name, scenario, time_years, step_pixels):
    """
    Computes a full synthetic temporal series for one scenario.
    """

    records = []
    solid_masks = []
    pore_masks = []

    for i, time_year in enumerate(time_years):
        iterations = i * step_pixels

        solid_mask = perturb_solid_mask(
            base_solid_mask,
            scenario=scenario,
            iterations=iterations
        )

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

        record = {
            "scenario": scenario,
            "base_file": base_name,
            "time_year": time_year,
            "iterations": iterations,

            "solid_fraction": solid["fraction"],
            "solid_largest_component_fraction": solid["largest_component_fraction"],
            "solid_interface_density": solid["interface_density"],
            "solid_euler_density": solid["euler_density"],

            "pore_fraction": pore["fraction"],
            "pore_area_mean_px": pore["area_mean_px"],
            "pore_largest_component_fraction": pore["largest_component_fraction"],
            "pore_interface_density": pore["interface_density"],
            "pore_euler_density": pore["euler_density"],
            "pore_percolates_x": pore["percolates_x"],
            "pore_percolates_y": pore["percolates_y"],
            "pore_anisotropy": pore["anisotropy"],

            "pore_tortuosity_x": tau_x,
            "pore_tortuosity_y": tau_y,
            "pore_tortuosity_effective": tau_eff,
        }

        records.append(record)
        solid_masks.append(solid_mask)
        pore_masks.append(pore_mask)

    df = pd.DataFrame(records)

    # ------------------------------------------------------------
    # GeoFriedmann scale factor
    # ------------------------------------------------------------

    pore_length = np.sqrt(df["pore_area_mean_px"].to_numpy(float))

    a_p = np.full(len(df), np.nan)
    valid = np.isfinite(pore_length) & (pore_length > 0)

    if np.any(valid):
        reference = pore_length[valid][0]
        a_p[valid] = pore_length[valid] / reference

    dt = float(time_years[1] - time_years[0])

    H_p = finite_difference(safe_log(a_p), dt=dt)
    da_p = finite_difference(a_p, dt=dt)
    d2a_p = finite_difference(da_p, dt=dt)

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
    # Reservoir potential
    # ------------------------------------------------------------

    pore_percolates = (
        df["pore_percolates_x"].astype(bool)
        | df["pore_percolates_y"].astype(bool)
    )

    connectivity_factor = (
        BASELINE_CONNECTIVITY
        + (1.0 - BASELINE_CONNECTIVITY) * pore_percolates.astype(float)
    )

    tau = df["pore_tortuosity_effective"].to_numpy(float)
    tau = np.where(
        np.isfinite(tau) & (tau > 0),
        tau,
        NON_PERCOLATING_TORTUOSITY
    )

    inverse_tortuosity = 1.0 / tau

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

    # ------------------------------------------------------------
    # Regime label
    # ------------------------------------------------------------

    regimes = []

    for _, row in df.iterrows():
        percolates = bool(row["pore_percolates_x"]) or bool(row["pore_percolates_y"])
        regimes.append(classify_regime(row["K_p"], percolates))

    df["regime"] = regimes

    return df, solid_masks, pore_masks


# ============================================================
# Plotting utilities
# ============================================================

def add_panel_label(ax, label):
    ax.text(
        -0.10,
        1.10,
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


def add_sequence_panel(ax, pore_masks, times, title):
    """
    Adds a row of synthetic pore masks inside one panel.
    """

    ax.axis("off")

    inset_positions = [
        [0.00, 0.10, 0.23, 0.78],
        [0.255, 0.10, 0.23, 0.78],
        [0.51, 0.10, 0.23, 0.78],
        [0.765, 0.10, 0.23, 0.78],
    ]

    for i, (mask, time_value) in enumerate(zip(pore_masks, times)):
        ax_in = ax.inset_axes(inset_positions[i])
        ax_in.imshow(mask, cmap="gray", interpolation="nearest")
        ax_in.set_xticks([])
        ax_in.set_yticks([])

        for spine in ax_in.spines.values():
            spine.set_linewidth(2)

        ax_in.set_title(
            f"{time_value} yr",
            fontsize=LABEL_SIZE,
            fontweight="bold",
            pad=6
        )

    ax.text(
        0.5,
        0.01,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )


def add_diagnostic_panel(ax, df, title, trend_note):
    """
    Adds descriptor trajectories for one synthetic scenario.
    """

    time_year = df["time_year"].to_numpy(float)

    ax.plot(
        time_year,
        df["pore_fraction"],
        marker="o",
        linewidth=3,
        markersize=10,
        label="Porosity"
    )

    ax.plot(
        time_year,
        df["R_res_norm"],
        marker="s",
        linewidth=3,
        markersize=9,
        linestyle="--",
        label=r"$R_{\mathrm{res}}$"
    )

    ax.set_xlabel(
        "Time (years)",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Porosity / normalized reservoir potential",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax.set_title(
        title,
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xticks(time_year)
    ax.grid(alpha=0.25)
    format_axis(ax)

    ax2 = ax.twinx()

    ax2.plot(
        time_year,
        df["K_p"],
        marker="D",
        linewidth=3,
        markersize=8,
        linestyle=":",
        label=r"$K_p$"
    )

    ax2.set_ylabel(
        r"Corrected curvature $K_p$",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax2.tick_params(axis="both", labelsize=TICK_SIZE, width=2, length=6)

    for tick in ax2.get_yticklabels():
        tick.set_fontweight("bold")

    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="best"
    )

    for _, row in df.iterrows():
        ax.text(
            row["time_year"],
            row["pore_fraction"] + 0.02,
            row["regime"],
            fontsize=11,
            fontweight="bold",
            ha="center"
        )

    ax.text(
        0.03,
        0.95,
        trend_note,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top"
    )


# ============================================================
# Figure generation
# ============================================================

def make_figure(open_df, open_masks, cement_df, cement_masks, base_file):
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    fig = plt.figure(figsize=FIGSIZE)

    gs = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.98,
        bottom=0.07,
        top=0.94,
        wspace=0.25,
        hspace=0.28
    )

    # Panel a
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "a")
    add_sequence_panel(
        ax_a,
        open_masks,
        TIME_YEARS,
        "Synthetic pore opening / dissolution"
    )

    # Panel b
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "b")
    add_diagnostic_panel(
        ax_b,
        open_df,
        "GeoFriedmann response: pore opening",
        r"Porosity $\uparrow$   $R_{\mathrm{res}}$ $\uparrow$   $K_p \downarrow$"
    )

    # Panel c
    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "c")
    add_sequence_panel(
        ax_c,
        cement_masks,
        TIME_YEARS,
        "Synthetic cementation / pore occlusion"
    )

    # Panel d
    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "d")
    add_diagnostic_panel(
        ax_d,
        cement_df,
        "GeoFriedmann response: cementation",
        r"Porosity $\downarrow$   $R_{\mathrm{res}}$ $\downarrow$   $K_p \uparrow$"
    )

    fig.suptitle(
        f"Synthetic Time Evolution of a Representative 2D Rock Lamina ({base_file})",
        fontsize=20,
        fontweight="bold",
        y=0.985
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

    if BASE_IMAGE_INDEX < 0 or BASE_IMAGE_INDEX >= len(image_paths):
        raise IndexError(
            f"BASE_IMAGE_INDEX={BASE_IMAGE_INDEX} is out of range for "
            f"{len(image_paths)} images."
        )

    base_path = image_paths[BASE_IMAGE_INDEX]
    base_solid_mask, detected_phase = read_solid_mask(base_path)

    print("Synthetic temporal evolution figure")
    print()
    print(f"Base image: {base_path.name}")
    print(f"Detected phase: {detected_phase}")
    print()

    step_pixels = default_step_pixels(base_solid_mask)

    print(f"Base perturbation step: {step_pixels} px")
    print(f"Times: {TIME_YEARS}")
    print()

    print("Computing synthetic pore opening / dissolution series...")
    open_df, open_solid_masks, open_pore_masks = compute_series(
        base_solid_mask=base_solid_mask,
        base_name=base_path.name,
        scenario="opening",
        time_years=TIME_YEARS,
        step_pixels=step_pixels
    )

    print("Computing synthetic cementation / occlusion series...")
    cement_df, cement_solid_masks, cement_pore_masks = compute_series(
        base_solid_mask=base_solid_mask,
        base_name=base_path.name,
        scenario="cementation",
        time_years=TIME_YEARS,
        step_pixels=step_pixels
    )

    all_df = pd.concat([open_df, cement_df], ignore_index=True)
    all_df.to_csv(OUTPUT_CSV, index=False)

    print("Generating figure...")
    make_figure(
        open_df=open_df,
        open_masks=open_pore_masks,
        cement_df=cement_df,
        cement_masks=cement_pore_masks,
        base_file=base_path.name
    )

    print()
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_SVG}")
    print(f"Saved: {OUTPUT_CSV}")

    print()
    print("Opening scenario summary:")
    print(
        open_df[
            [
                "time_year",
                "pore_fraction",
                "pore_largest_component_fraction",
                "pore_tortuosity_effective",
                "a_p",
                "H_p",
                "K_p",
                "R_res",
                "R_res_norm",
                "regime"
            ]
        ].to_string(index=False)
    )

    print()
    print("Cementation scenario summary:")
    print(
        cement_df[
            [
                "time_year",
                "pore_fraction",
                "pore_largest_component_fraction",
                "pore_tortuosity_effective",
                "a_p",
                "H_p",
                "K_p",
                "R_res",
                "R_res_norm",
                "regime"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
