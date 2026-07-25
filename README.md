# GeoFriedmann

Reproducibility repository for the manuscript:

**A GeoFriedmann Framework for Image-Based Pore-Space Evolution and Reservoir-Quality Screening in 2D Rock Laminae**

## Overview

GeoFriedmann is a reduced-order computational framework for comparing segmented two-dimensional pore structures using geometric, topological, percolation, anisotropy, and shortest-path descriptors.

This repository contains the three structures analyzed in the manuscript and the Python scripts used to generate Figures 1–7.

The GeoFriedmann coordinates and screening scores are image-derived, reference-set-dependent descriptors. They should not be interpreted as physical evolution laws, absolute permeability estimates, or predictions of reservoir productivity.

## Repository contents

- `structures.tar.gz`: archive containing the three original structures used in the analysis.
- `figure1/`: Python script used to generate Figure 1.
- `figure2/`: Python script used to generate Figure 2.
- `figure3/`: Python script used to generate Figure 3.
- `figure4/`: Python script used to generate Figure 4.
- `figure5/`: Python script used to generate Figure 5.
- `figure6/`: Python script used to generate Figure 6.
- `figure7/`: Python script used to generate Figure 7.
- `requirements.txt`: Python dependencies used to reproduce the results.

## Installation

Clone the repository:

```bash
git clone https://github.com/tromer-unb/GeoFriedmann.git
cd GeoFriedmann
