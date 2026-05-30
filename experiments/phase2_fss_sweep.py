import os
import json
import math
import numpy as np
import random
import multiprocessing
from typing import Dict, List, Any
from src.geometry.spacetime import SpacetimeComplex
from src.syndrome.noise import NoiseModel
from src.enumeration.scaling_engine import ScalingDPEngine
from src.persistence.filtration import FiltrationEngine
from src.persistence.homology import HomologyEngine
from src.decoding.baseline import MWPMBaselineDecoder

def run_single_fss_trial(args):
    """
    Worker function to process a single FSS Monte Carlo trial.
    """
    d, p, seed, trial = args
    # Ensure independent random seeds per process
    np.random.seed(seed + trial)
    random.seed(seed + trial)
    
    complex_lattice = SpacetimeComplex(d=d, T=d)
    
    # Subsystem engine allocation setup
    noise = NoiseModel(complex_lattice, p_spatial=p, p_temporal=p)
    mwpm = MWPMBaselineDecoder(complex_lattice, p_spatial=p, p_temporal=p)
    
    # Dynamically select DP Engine based on lattice size
    if d == 3:
        from src.enumeration.kernel import SpatialKernelCache
        from src.enumeration.dp_engine import ExactDPEngine
        cache = SpatialKernelCache(d=d, p_spatial=p)
        dp_engine = ExactDPEngine(complex_lattice, kernel_cache=cache, p_temporal=p)
    else:
        cutoff = 1e-5 if p <= 0.04 else 1e-3
        dp_engine = ScalingDPEngine(complex_lattice, p_spatial=p, p_temporal=p, k_max=2, epsilon_cutoff=cutoff)
    
    # 1. Native physical noise instantiation
    faults = noise.sample_error_configuration()
    s_obs = noise.compute_syndrome(faults)
    
    # 2. Classical Baseline decoder profiling
    fail_x, fail_y = mwpm.decode_syndrome(s_obs, faults)
    is_mwpm_fail = int(fail_x or fail_y)
        
    # 3. Dynamic Forward-Backward Sparse Partition Contractions
    try:
        marginals = dp_engine.compute_edge_marginals(s_obs)
    except Exception:
        # Catch boundary initialization traps or pruning errors
        return is_mwpm_fail, None
        
    # 4. Filtration construction mapping Bayesian marginal distributions using FiltrationEngine
    filt = FiltrationEngine(complex_lattice, default_bg_prob=1e-5)
    cap_weight = -math.log(1e-5)
    
    for edge_data in complex_lattice.all_edges():
        u, v = edge_data['vertices']
        c_edge = tuple(sorted((u, v)))
        prob = marginals.get(c_edge, 1e-6)
        prob = min(0.9999, max(1e-6, prob))
        filt.edge_weights[c_edge] = min(cap_weight, -math.log(prob))
        
    export_data = filt.export_filtration_data(cone_weight=-1.0)
    export_data['max_weight'] = cap_weight
        
    # 5. GUDHI persistent relative loop lifetime parameter evaluations
    homology = HomologyEngine(complex_lattice, export_data)
    obs = homology.compute_observables(persistence_threshold=0.5, cap_infinity=cap_weight)
    return is_mwpm_fail, obs['L_logical']

def run_finite_size_scaling_sweep(distances: List[int], p_range: List[float], 
                                  num_trials: int = 200, output_path: str = "results/phase2_fss_data.json",
                                  seed: int = 42):
    r"""
    Orchestrates the multi-distance Finite-Size Scaling (FSS) parameter sweep.
    Outputs highly structured, persistent JSON maps ready for FSS data collapse analysis.
    Uses multiprocessing to accelerate execution.
    """
    results: Dict[str, Any] = {
        "metadata": {
            "distances": distances,
            "num_trials_target": num_trials,
            "seed": seed,
            "filtration_threshold": 0.5,
            "engine": "MixedDPEngine"
        },
        "points": {}
    }
    
    # Load existing results if they exist (checkpointing)
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                existing_data = json.load(f)
                if existing_data.get("metadata", {}).get("seed") == seed:
                    results = existing_data
                    print(f"🔄 Resuming from existing checkpoint: {len(results['points'])} distances loaded.")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}. Starting fresh.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("==================================================================")
    print(f"🌌 Initiating Multi-Distance FSS Sweep Engine across d in {distances} (PARALLEL)")
    print(f" -> Target Fault Baselines: {p_range}")
    print(f" -> Monte Carlo Batching:   {num_trials} evaluations per sample point")
    print("==================================================================")

    # Use all but 2 cores, min 1
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Utilizing {num_cores} parallel workers...")

    for d in distances:
        d_str = f"d={d}"
        if d_str not in results["points"]:
            results["points"][d_str] = {}
            
        print(f"\n=================== Lattice Dimension: d = {d} ===================")
        
        for p in p_range:
            p_str = f"{p:.4f}"
            if p_str in results["points"][d_str]:
                print(f"⏭️ Skipping d = {d}, p = {p_str} (already completed).")
                continue
                
            print(f"\n--- Scanning Physical Base Rate: p = {p_str} ---")
            
            # Build task arguments
            task_args = [(d, p, seed, trial) for trial in range(num_trials)]
            
            failures_mwpm = 0
            lifetimes_logical = []

            with multiprocessing.Pool(num_cores) as pool:
                completed = 0
                for is_mwpm_fail, L_logical in pool.imap_unordered(run_single_fss_trial, task_args):
                    if is_mwpm_fail:
                        failures_mwpm += 1
                    if L_logical is not None:
                        lifetimes_logical.append(L_logical)
                    
                    completed += 1
                    if completed % 50 == 0:
                        print(f"    Progress: {completed}/{num_trials} trials completed...", flush=True)

            p_fail_mwpm = failures_mwpm / num_trials
            avg_l_logical = float(np.mean(lifetimes_logical)) if lifetimes_logical else 0.0
            std_l_logical = float(np.std(lifetimes_logical)) if lifetimes_logical else 0.0
            var_l_logical = float(np.var(lifetimes_logical)) if lifetimes_logical else 0.0
            
            results["points"][d_str][p_str] = {
                'p_physical': p,
                'p_fail_mwpm': p_fail_mwpm,
                'avg_L_logical': avg_l_logical,
                'std_L_logical': std_l_logical,
                'var_L_logical': var_l_logical,
                'trials_evaluated': len(lifetimes_logical)
            }
            
            # Atomic Checkpoint Save
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f" -> Metrics: P_fail(MWPM) = {p_fail_mwpm:.3f} | <L_logical> = {avg_l_logical:.3f} ± {std_l_logical:.3f} (evaluated {len(lifetimes_logical)} trials)")

    print("\n==================================================================")
    print(f"✅ Multi-Distance FSS Sweep successfully delivered to:\n -> {output_path}")
    print("==================================================================")
    return results

if __name__ == '__main__':
    # Standard FSS evaluation execution configuration
    target_distances = [3, 5]
    critical_p_range = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    run_finite_size_scaling_sweep(target_distances, critical_p_range, num_trials=200)
