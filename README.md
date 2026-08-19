# mhec
High-temperature elastic constants from MLFF-accelerated AIMD for all crystal systems
# M-HEC

M-HEC computes finite-temperature single-crystal elastic constants by combining
a machine-learning force field with first-principles molecular dynamics. It
covers all seven crystal systems, offers two complementary deformation modes,
and reconstructs the full stiffness tensor with Born-stability and
residual-pressure corrections.

## Features
- Automatic crystal-system and Laue-class detection from the space group
- Single-component (SC-SS) and symmetry-adapted (SA-SS) deformation modes
- MLFF trained and validated on the same strain domain probed in the fit
- VRH moduli, anisotropy, Debye temperature, and sound velocities

## Requirements
Python 3.7 or later, NumPy, SciPy. Matplotlib is optional for plotting.
VASP 6.3 or later serves as the DFT and MD engine.

## Installation
pip install -e .

## Usage
Run the command-line interface and follow the menu.

## License
MIT
