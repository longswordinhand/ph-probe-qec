# Persistent Homology as a Stable Multiscale Descriptor of QEC Spacetime Complex Errors

This repository contains the source code, data files, and LaTeX manuscript for the persistent homology analysis of topological quantum error correction (QEC) on (2+1)-dimensional spacetime complexes.

## Repository Structure

| Path | Description |
|---|---|
| `src/` | Core Python packages for decoding engines, filtrations, persistent homology geometry, and syndromes. |
| `experiments/` | Python scripts for running Monte Carlo simulations and plotting results. |
| `manuscript/` | LaTeX source documents (`main.tex`, `references.bib`), compilation python scripts, and output PDF. |
| `results/` | Numeric dataset outputs (`phase2_fss_data.json`, `dense_d3_sweep_data.json`) and generated vector figures. |
| `tests/` | Pytest test suite for validating all underlying decoding and dynamic programming engines. |

## Quick Start & Environment Setup

This project uses a project-local virtual environment (`.venv`). Follow these steps to set up and run the code locally:

```bash
# Clone the repository
git clone https://github.com/longswordinhand/ph-probe-qec.git
cd ph-probe-qec

# Activate the local virtual environment
source .venv/bin/activate

# Install any required dependencies if not already present
pip install -r requirements.txt # (if needed)
```

## Running Simulations & Generating Figures

To regenerate the multi-distance threshold comparison plot and topological susceptibility plots from the archived numerical data:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the plotting script
python experiments/plot_fss_results.py
```
Output figures will be saved in the `results/` directory as `multi_distance_threshold_comparison.pdf` and `topological_susceptibility.pdf`.

## Compiling the LaTeX Manuscript

The manuscript can be compiled to PDF directly using the provided build script:

```bash
python manuscript/compile.py
```
This script prepends local TinyTeX paths (if available), regenerates the figures, runs `pdflatex` and `bibtex` in a multi-pass compilation loop, and outputs the publication-ready PDF to `manuscript/main.pdf`.

## Running the Tests

To run the full test suite verifying the dynamic programming engines and persistent homology filtrations:

```bash
PYTHONPATH=. pytest tests/
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
