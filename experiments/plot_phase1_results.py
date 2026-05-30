import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def render_comprehensive_plots(data_path: str, output_dir: str):
    """
    Renders publication-quality plots including:
    1. Threshold Comparison with Variance Shading
    2. Topological Susceptibility (Variance of L_logical)
    Supports partial data from checkpoints.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both old flat format and new structured format
    points = data.get("points", data)
    metadata = data.get("metadata", {})
    
    # Sort points by physical error rate ascending
    sorted_items = sorted(points.values(), key=lambda x: x["p_physical"])
    
    p_phys = np.array([item["p_physical"] for item in sorted_items])
    p_fail_mwpm = np.array([item["p_fail_mwpm"] for item in sorted_items])
    L_logical = np.array([item["avg_L_logical"] for item in sorted_items])
    L_std = np.array([item.get("std_L_logical", 0.0) for item in sorted_items])
    L_var = np.array([item.get("var_L_logical", L_std**2) for item in sorted_items])

    # Apply premium journal-ready formatting
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('default')

    # --- Plot 1: Primary Threshold Comparison ---
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
    color_dp = "#0d9488"   # Sleek Teal
    color_mwpm = "#e11d48" # Rose Red

    ax1.plot(p_phys, L_logical, marker='o', markersize=8, linewidth=2.5, 
            color=color_dp, label=r'Exact DP Posterior Homology $\langle L_{\mathrm{logical}} \rangle$', zorder=4)
    
    if np.any(L_std > 0):
        ax1.fill_between(p_phys, np.maximum(0, L_logical - L_std), L_logical + L_std, 
                        color=color_dp, alpha=0.2, zorder=3, 
                        label=r'$\pm 1\sigma$ Topological Fluctuation')

    ax1.plot(p_phys, p_fail_mwpm, marker='s', markersize=8, linewidth=2.5, linestyle='--',
            color=color_mwpm, label=r'PyMatching MWPM Baseline ($P_{\mathrm{fail}}$)', zorder=5)

    ax1.set_xscale('log')
    ax1.set_xlabel(r'Physical Error Rate ($p$)', fontsize=13, fontweight='bold')
    ax1.set_ylabel(r'Logical Breakdown / Decoding Failure Rate', fontsize=13, fontweight='bold')
    ax1.set_title(r'Surface Code QEC: Precursor Topological Stability', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left')
    
    # --- Plot 2: Topological Susceptibility (Fluctuation Growth) ---
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)
    ax2.plot(p_phys, L_var, marker='D', markersize=7, linewidth=2, color="#7c3aed", # Violet
            label=r'Topological Susceptibility $\chi = \mathrm{Var}(L_{\mathrm{logical}})$')
    
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Physical Error Rate ($p$)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'Variance $\chi$', fontsize=12, fontweight='bold')
    ax2.set_title(r'Early-Warning Signal: Fluctuation Growth near Threshold', fontsize=13, fontweight='bold')
    ax2.legend(frameon=True, facecolor='white', framealpha=0.95)

    # Save Results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    fig1.tight_layout()
    fig1.savefig(out_path / "dense_d3_threshold_comparison.pdf")
    fig1.savefig(out_path / "dense_d3_threshold_comparison.png")
    
    fig2.tight_layout()
    fig2.savefig(out_path / "topological_susceptibility.pdf")
    fig2.savefig(out_path / "topological_susceptibility.png")
    
    plt.close('all')
    print(f"📈 Comprehensive plots rendered to: {output_dir}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_json = base_dir / "results" / "dense_d3_sweep_data.json"
    out_directory = base_dir / "results"
    render_comprehensive_plots(str(data_json), str(out_directory))
