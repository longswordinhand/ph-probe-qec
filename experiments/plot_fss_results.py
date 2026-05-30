import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def render_fss_plots(data_path: str, output_dir: str):
    """
    Renders publication-quality FSS plots including:
    1. Threshold Crossing: L_logical and P_fail for both d=3 and d=5
    2. Topological Susceptibility (Variance of L_logical) for d=3 and d=5
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = data.get("points", {})
    
    # Extract distances
    distances = sorted([int(k.split("=")[1]) for k in points.keys()])
    print(f"📊 Found data for distances: {distances}")

    # Set up matplotlib style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('default')

    # Color palettes
    colors_logical = {3: "#0d9488", 5: "#1d4ed8"}  # Teal (d=3), Blue (d=5)
    colors_mwpm = {3: "#f97316", 5: "#be123c"}     # Orange (d=3), Rose Red (d=5)
    colors_suscep = {3: "#7c3aed", 5: "#db2777"}    # Violet (d=3), Pink (d=5)
    markers = {3: "o", 5: "^"}
    mwpm_markers = {3: "s", 5: "D"}

    # --- Plot 1: Threshold Comparison ---
    fig1, ax1 = plt.subplots(figsize=(8.5, 6.5), dpi=300)

    for d in distances:
        d_key = f"d={d}"
        if d_key not in points:
            continue
        
        sorted_items = sorted(points[d_key].items(), key=lambda x: float(x[0]))
        p_phys = np.array([float(k) for k, _ in sorted_items])
        p_fail_mwpm = np.array([item["p_fail_mwpm"] for _, item in sorted_items])
        L_logical = np.array([item["avg_L_logical"] for _, item in sorted_items])
        L_std = np.array([item.get("std_L_logical", 0.0) for _, item in sorted_items])

        # Plot exact DP posterior homology logical breakdown
        ax1.plot(p_phys, L_logical, marker=markers[d], markersize=8, linewidth=2.5,
                 color=colors_logical[d], label=rf'Exact DP Posterior $\langle L_{{\mathrm{{logical}}}} \rangle$ ($d={d}$)', zorder=4)
        
        # Plot standard deviation shading
        if np.any(L_std > 0):
            ax1.fill_between(p_phys, np.maximum(0, L_logical - L_std), L_logical + L_std,
                             color=colors_logical[d], alpha=0.15, zorder=3,
                             label=rf'$\pm 1\sigma$ Topological Fluctuation ($d={d}$)')

        # Plot MWPM baseline
        ax1.plot(p_phys, p_fail_mwpm, marker=mwpm_markers[d], markersize=8, linewidth=2.0, linestyle='--',
                 color=colors_mwpm[d], label=rf'MWPM Decoder $P_{{\mathrm{{fail}}}}$ ($d={d}$)', zorder=5)

    ax1.set_xscale('log')
    ax1.set_xlabel(r'Physical Error Rate ($p$)', fontsize=14, fontweight='bold')
    ax1.set_ylabel(r'Logical Breakdown / Decoding Failure Rate', fontsize=14, fontweight='bold')
    ax1.set_title(r'Surface Code QEC: Multi-Distance Topological Stability', fontsize=15, fontweight='bold', pad=15)
    
    # Clean grid and legend
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left', fontsize=10)
    
    # --- Plot 2: Topological Susceptibility ---
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    for d in distances:
        d_key = f"d={d}"
        if d_key not in points:
            continue
        
        sorted_items = sorted(points[d_key].items(), key=lambda x: float(x[0]))
        p_phys = np.array([float(k) for k, _ in sorted_items])
        L_var = np.array([item.get("var_L_logical", 0.0) for _, item in sorted_items])

        # Plot susceptibility
        ax2.plot(p_phys, L_var, marker=markers[d], markersize=8, linewidth=2.5,
                 color=colors_suscep[d], label=rf'Topological Susceptibility $\chi$ ($d={d}$)')

    ax2.set_xscale('log')
    ax2.set_xlabel(r'Physical Error Rate ($p$)', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'Susceptibility $\chi = \mathrm{Var}(L_{\mathrm{logical}})$', fontsize=13, fontweight='bold')
    ax2.set_title(r'Topological Susceptibility Across Physical Error Rates', fontsize=14, fontweight='bold', pad=15)
    
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left', fontsize=11)

    # Save outputs
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    fig1.tight_layout()
    fig1.savefig(out_path / "multi_distance_threshold_comparison.pdf")
    fig1.savefig(out_path / "multi_distance_threshold_comparison.png")
    
    fig2.tight_layout()
    fig2.savefig(out_path / "topological_susceptibility.pdf")
    fig2.savefig(out_path / "topological_susceptibility.png")
    
    plt.close('all')
    print(f"📈 Comprehensive FSS plots successfully rendered to: {output_dir}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_json = base_dir / "results" / "phase2_fss_data.json"
    out_directory = base_dir / "results"
    render_fss_plots(str(data_json), str(out_directory))
