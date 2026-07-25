#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5 generator for the GeoFriedmann 2D rock laminae paper.

This figure evaluates the robustness of the GeoFriedmann reservoir
ranking under segmentation perturbation, image-scale perturbation,
and reservoir-index component ablation.

Panels:
    a) Segmentation sensitivity of reservoir potential
    b) Scale sensitivity of corrected pore curvature
    c) Ablation matrix of reservoir-index components
    d) Robustness envelope of reservoir potential

Inputs:
    structures/

Outputs:
    figure5_robustness_ablation.png
    figure5_robustness_ablation.pdf
    figure5_robustness_ablation.svg
    figure5_robustness_ablation.csv
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

OUTPUT_PNG = "figure5_robustness_ablation.png"
OUTPUT_PDF = "figure5_robustness_ablation.pdf"
OUTPUT_SVG = "figure5_robustness_ablation.svg"
OUTPUT_CSV = "figure5_robustness_ablation.csv"

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

MAX_TORTUOSITY_SIZE = 700

SEGMENTATION_PERTURBATIONS = [-2, -1, 0, 1, 2]
SCALE_FACTORS = [1, 2, 4, 8]

BASELINE_CONNECTIVITY = 0.25
NON_PERCOLATING_TORTUOSITY = 4.0


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


def load_base_masks(image_paths):
    masks = []
    phases = []

    for path in image_paths:
        mask, detected_phase = read_solid_mask(path)
        masks.append(mask)
        phases.append(detected_phase)

    return masks, phases


# ============================================================
# Mask perturbation
# ============================================================

def perturb_solid_mask(solid_mask, radius):
    """
    Applies a controlled segmentation perturbation.

    radius < 0:
        erodes the solid phase, increasing pore space.

    radius > 0:
        dilates the solid phase, decreasing pore space.

    radius = 0:
        returns the original mask.
    """

    solid_mask = solid_mask.astype(bool)

    if radius == 0:
        return solid_mask.copy()

    structure = np.ones((3, 3), dtype=bool)

    if radius > 0:
        return ndi.binary_dilation(
            solid_mask,
            structure=structure,
            iterations=int(radius)
        )

    return ndi.binary_erosion(
        solid_mask,
        structure=structure,
        iterations=int(abs(radius))
    )


def downsample_mask(mask, factor):
    """
    Downsamples a binary mask using simple striding.

    factor = 1 returns the original mask.
    """

    mask = mask.astype(bool)

    if factor <= 1:
        return mask.copy()

    return mask[::factor, ::factor].astype(bool)


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


# ============================================================
# Experiment computation
# ============================================================

def compute_stage_descriptors(
    solid_masks,
    image_paths,
    experiment_family,
    perturbation_value,
    perturbation_label
):
    """
    Computes descriptors for a full set of stages under one perturbation.
    """

    rows = []

    for stage, solid_mask in enumerate(solid_masks, start=1):
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
            "experiment_family": experiment_family,
            "perturbation_label": perturbation_label,
            "perturbation_value": perturbation_value,

            "stage": stage,
            "file": image_paths[stage - 1].name,

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
    df = add_curvature_and_reservoir_index(df)

    return df


def add_curvature_and_reservoir_index(
    df,
    use_porosity=True,
    use_connectivity=True,
    use_largest_network=True,
    use_tortuosity=True,
    use_anisotropy=True
):
    """
    Adds corrected pore curvature and reservoir potential.

    Base reservoir index:

        R_res = phi * C_p * L_p * tau^{-1} * (1 + A_p)
    """

    df = df.copy()

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

    pore_percolates = (
        df["pore_percolates_x"].astype(bool)
        | df["pore_percolates_y"].astype(bool)
    )

    if use_porosity:
        phi = df["pore_fraction"].to_numpy(float)
    else:
        phi = np.ones(len(df), dtype=float)

    if use_connectivity:
        connectivity_factor = (
            BASELINE_CONNECTIVITY
            + (1.0 - BASELINE_CONNECTIVITY) * pore_percolates.astype(float)
        )
    else:
        connectivity_factor = np.ones(len(df), dtype=float)

    if use_largest_network:
        largest_network = df["pore_largest_component_fraction"].to_numpy(float)
    else:
        largest_network = np.ones(len(df), dtype=float)

    tau = df["pore_tortuosity_effective"].to_numpy(float)

    tau = np.where(
        np.isfinite(tau) & (tau > 0),
        tau,
        NON_PERCOLATING_TORTUOSITY
    )

    if use_tortuosity:
        inverse_tortuosity = 1.0 / tau
    else:
        inverse_tortuosity = np.ones(len(df), dtype=float)

    if use_anisotropy:
        anisotropy_factor = 1.0 + df["pore_anisotropy"].fillna(0).to_numpy(float)
    else:
        anisotropy_factor = np.ones(len(df), dtype=float)

    R_res = (
        phi
        * connectivity_factor
        * largest_network
        * inverse_tortuosity
        * anisotropy_factor
    )

    df["pore_connectivity_factor"] = connectivity_factor
    df["inverse_tortuosity"] = inverse_tortuosity
    df["R_res"] = R_res
    df["R_res_norm_local"] = minmax(R_res)

    return df


def compute_segmentation_robustness(base_masks, image_paths):
    records = []

    for radius in SEGMENTATION_PERTURBATIONS:
        print(f"Segmentation perturbation: {radius:+d} px")

        perturbed_masks = [
            perturb_solid_mask(mask, radius)
            for mask in base_masks
        ]

        df = compute_stage_descriptors(
            perturbed_masks,
            image_paths,
            experiment_family="segmentation",
            perturbation_value=radius,
            perturbation_label=f"{radius:+d} px"
        )

        records.append(df)

    return pd.concat(records, ignore_index=True)


def compute_scale_robustness(base_masks, image_paths):
    records = []

    for factor in SCALE_FACTORS:
        print(f"Scale factor: 1/{factor}")

        scaled_masks = [
            downsample_mask(mask, factor)
            for mask in base_masks
        ]

        df = compute_stage_descriptors(
            scaled_masks,
            image_paths,
            experiment_family="scale",
            perturbation_value=factor,
            perturbation_label=f"1/{factor}"
        )

        records.append(df)

    return pd.concat(records, ignore_index=True)


def compute_ablation(base_masks, image_paths):
    print("Computing ablation study")

    base_df = compute_stage_descriptors(
        base_masks,
        image_paths,
        experiment_family="ablation",
        perturbation_value=0,
        perturbation_label="base"
    )

    variants = [
        {
            "name": "Full",
            "use_porosity": True,
            "use_connectivity": True,
            "use_largest_network": True,
            "use_tortuosity": True,
            "use_anisotropy": True,
        },
        {
            "name": "No porosity",
            "use_porosity": False,
            "use_connectivity": True,
            "use_largest_network": True,
            "use_tortuosity": True,
            "use_anisotropy": True,
        },
        {
            "name": "No connectivity",
            "use_porosity": True,
            "use_connectivity": False,
            "use_largest_network": True,
            "use_tortuosity": True,
            "use_anisotropy": True,
        },
        {
            "name": "No network",
            "use_porosity": True,
            "use_connectivity": True,
            "use_largest_network": False,
            "use_tortuosity": True,
            "use_anisotropy": True,
        },
        {
            "name": "No tortuosity",
            "use_porosity": True,
            "use_connectivity": True,
            "use_largest_network": True,
            "use_tortuosity": False,
            "use_anisotropy": True,
        },
        {
            "name": "No anisotropy",
            "use_porosity": True,
            "use_connectivity": True,
            "use_largest_network": True,
            "use_tortuosity": True,
            "use_anisotropy": False,
        },
    ]

    records = []

    for variant in variants:
        df_variant = add_curvature_and_reservoir_index(
            base_df,
            use_porosity=variant["use_porosity"],
            use_connectivity=variant["use_connectivity"],
            use_largest_network=variant["use_largest_network"],
            use_tortuosity=variant["use_tortuosity"],
            use_anisotropy=variant["use_anisotropy"],
        )

        df_variant["experiment_family"] = "ablation"
        df_variant["perturbation_label"] = variant["name"]
        df_variant["perturbation_value"] = np.nan
        df_variant["ablation_variant"] = variant["name"]

        records.append(df_variant)

    ablation_df = pd.concat(records, ignore_index=True)

    ablation_df["R_res_ablation_norm"] = minmax(
        ablation_df["R_res"].to_numpy(float)
    )

    return ablation_df


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

def make_figure(seg_df, scale_df, ablation_df):
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    robust_df = pd.concat([seg_df, scale_df], ignore_index=True)
    robust_df["R_res_global_norm"] = minmax(
        robust_df["R_res"].to_numpy(float)
    )

    seg_df = robust_df[robust_df["experiment_family"] == "segmentation"].copy()
    scale_df = robust_df[robust_df["experiment_family"] == "scale"].copy()

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

    stages = sorted(robust_df["stage"].unique())

    # ============================================================
    # Panel a: segmentation sensitivity
    # ============================================================

    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "a")

    for stage in stages:
        sub = seg_df[seg_df["stage"] == stage].sort_values("perturbation_value")

        ax_a.plot(
            sub["perturbation_value"],
            sub["R_res_global_norm"],
            marker="o",
            linewidth=3,
            markersize=10,
            label=f"Stage {int(stage)}"
        )

    ax_a.axvline(0, linewidth=2, linestyle="--", alpha=0.7)

    ax_a.set_xlabel(
        "Segmentation perturbation (px)",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_a.set_ylabel(
        r"Normalized $R_{\mathrm{res}}$",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_a.set_title(
        "Segmentation Sensitivity",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax_a.set_xticks(SEGMENTATION_PERTURBATIONS)
    ax_a.set_ylim(-0.05, 1.08)
    ax_a.grid(alpha=0.25)
    ax_a.legend(frameon=False, fontsize=LEGEND_SIZE)
    format_axis(ax_a)

    # ============================================================
    # Panel b: scale sensitivity
    # ============================================================

    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "b")

    for stage in stages:
        sub = scale_df[scale_df["stage"] == stage].sort_values("perturbation_value")

        ax_b.plot(
            sub["perturbation_value"],
            sub["K_p"],
            marker="s",
            linewidth=3,
            markersize=10,
            label=f"Stage {int(stage)}"
        )

    ax_b.axhline(0, linewidth=2, linestyle="--", alpha=0.7)

    ax_b.set_xlabel(
        "Downsampling factor",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_b.set_ylabel(
        r"Corrected pore curvature $K_p$",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_b.set_title(
        "Scale Sensitivity",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax_b.set_xticks(SCALE_FACTORS)
    ax_b.grid(alpha=0.25)
    ax_b.legend(frameon=False, fontsize=LEGEND_SIZE)
    format_axis(ax_b)

    # ============================================================
    # Panel c: ablation matrix
    # ============================================================

    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "c")

    variant_order = [
        "Full",
        "No porosity",
        "No connectivity",
        "No network",
        "No tortuosity",
        "No anisotropy",
    ]

    matrix = []

    for variant in variant_order:
        row_values = []

        for stage in stages:
            value = ablation_df[
                (ablation_df["ablation_variant"] == variant)
                & (ablation_df["stage"] == stage)
            ]["R_res_ablation_norm"].values

            if len(value) == 0:
                row_values.append(np.nan)
            else:
                row_values.append(value[0])

        matrix.append(row_values)

    matrix = np.asarray(matrix, dtype=float)

    im = ax_c.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=1
    )

    ax_c.set_xticks(np.arange(len(stages)))
    ax_c.set_xticklabels([str(int(s)) for s in stages])

    ax_c.set_yticks(np.arange(len(variant_order)))
    ax_c.set_yticklabels(variant_order)

    ax_c.set_xlabel(
        "Stage",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_c.set_title(
        "Reservoir-Index Ablation",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax_c.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold"
                )

    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized score", fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=13, width=2)

    format_axis(ax_c)

    # ============================================================
    # Panel d: robustness envelope
    # ============================================================

    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "d")

    data = []

    for stage in stages:
        values = robust_df[
            robust_df["stage"] == stage
        ]["R_res_global_norm"].to_numpy(float)

        values = values[np.isfinite(values)]
        data.append(values)

    positions = np.arange(1, len(stages) + 1)

    box = ax_d.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False
    )

    for patch in box["boxes"]:
        patch.set_linewidth(2)
        patch.set_alpha(0.65)

    for element in ["whiskers", "caps", "medians"]:
        for item in box[element]:
            item.set_linewidth(2)

    rng = np.random.default_rng(123)

    for i, values in enumerate(data):
        jitter = rng.normal(0, 0.035, size=len(values))

        ax_d.scatter(
            np.full(len(values), positions[i]) + jitter,
            values,
            s=80,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.9,
            zorder=3
        )

    ax_d.set_xticks(positions)
    ax_d.set_xticklabels([str(int(s)) for s in stages])

    ax_d.set_xlabel(
        "Stage",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_d.set_ylabel(
        r"Robust $R_{\mathrm{res}}$ envelope",
        fontsize=LABEL_SIZE,
        fontweight="bold"
    )

    ax_d.set_title(
        "Robustness Envelope",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax_d.set_ylim(-0.05, 1.08)
    ax_d.grid(axis="y", alpha=0.25)
    format_axis(ax_d)

    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")

    plt.close(fig)

    robust_df["record_group"] = "robustness"
    ablation_df["record_group"] = "ablation"

    combined = pd.concat(
        [robust_df, ablation_df],
        ignore_index=True,
        sort=False
    )

    combined.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# Main
# ============================================================

def main():
    image_paths = list_image_files(INPUT_DIR)

    print("Images used for Figure 5:")

    for path in image_paths:
        print(f"  - {path.name}")

    print()
    print("Loading base masks...")
    base_masks, phases = load_base_masks(image_paths)

    print()
    print("Computing segmentation robustness...")
    seg_df = compute_segmentation_robustness(base_masks, image_paths)

    print()
    print("Computing scale robustness...")
    scale_df = compute_scale_robustness(base_masks, image_paths)

    print()
    print("Computing ablation analysis...")
    ablation_df = compute_ablation(base_masks, image_paths)

    print()
    print("Generating Figure 5...")
    make_figure(seg_df, scale_df, ablation_df)

    print()
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_SVG}")
    print(f"Saved: {OUTPUT_CSV}")

    print()
    print("Ablation summary:")

    summary = ablation_df[
        ablation_df["ablation_variant"] == "Full"
    ][
        [
            "stage",
            "file",
            "pore_fraction",
            "pore_largest_component_fraction",
            "pore_tortuosity_effective",
            "K_p",
            "R_res",
            "R_res_norm_local",
        ]
    ]

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
