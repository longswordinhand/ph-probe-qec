import os
import json
import math
import numpy as np
import random
import multiprocessing
from typing import Dict, List, Any
from src.geometry.spacetime import SpacetimeComplex
from src.syndrome.noise import NoiseModel
from src.enumeration.kernel import SpatialKernelCache
from src.enumeration.dp_engine import ExactDPEngine
from src.persistence.filtration import FiltrationEngine
from src.persistence.homology import HomologyEngine
from src.decoding.baseline import MWPMBaselineDecoder

def run_single_trial(args):
    """
    Worker function to process a single Monte Carlo trial.
    Must be defined at the top-level of the module for pickling.
    """
    p, seed, trial = args
    # Ensure independent random seeds per process
    np.random.seed(seed + trial)
    random.seed(seed + trial)
    
    d = 3
    complex_lattice = SpacetimeComplex(d=d, T=d)
    
    noise = NoiseModel(complex_lattice, p_spatial=p, p_temporal=p)
    kernel = SpatialKernelCache(d=d, p_spatial=p)
    dp_engine = ExactDPEngine(complex_lattice, kernel_cache=kernel, p_temporal=p)
    mwpm = MWPMBaselineDecoder(complex_lattice, p_spatial=p, p_temporal=p)
    
    faults = noise.sample_error_configuration()
    s_obs = noise.compute_syndrome(faults)
    
    # 1. Classical Baseline
    fail_x, fail_y = mwpm.decode_syndrome(s_obs, faults)
    is_mwpm_fail = int(fail_x or fail_y)
    
    # 2. Exact Bayesian DP Inference
    try:
        marginals = dp_engine.compute_edge_marginals(s_obs)
    except Exception:
        return is_mwpm_fail, None
        
    # 3. Filtration Embedding using FiltrationEngine
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
        
    # 4. Topological Feature Extraction
    homology = HomologyEngine(complex_lattice, export_data)
    obs = homology.compute_observables(persistence_threshold=0.5, cap_infinity=cap_weight)
    return is_mwpm_fail, obs['L_logical']

def run_dense_d3_sweep(p_range: List[float], num_trials: int = 500, 
                       output_path: str = "results/dense_d3_sweep_data.json",
                       seed: int = 42):
    r"""
    Executes a high-density, statistically significant parameter sweep exclusively on d=3 Spacetime Complex.
    Includes checkpointing per noise point and metadata for reproducibility.
    Uses multiprocessing to accelerate executions.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    results: Dict[str, Any] = {
        "metadata": {
            "distance": 3,
            "T": 3,
            "num_trials_target": num_trials,
            "seed": seed,
            "filtration_threshold": 0.5,
            "engine": "ExactDPEngine"
        },
        "points": {}
    }
    
    # Load existing results if they exist (checkpointing)
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                existing_data = json.load(f)
                # Ensure we are resuming a compatible run
                if existing_data.get("metadata", {}).get("seed") == seed:
                    results = existing_data
                    print(f"🔄 Resuming from existing checkpoint: {len(results['points'])} points completed.")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}. Starting fresh.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("==================================================================")
    print(f"🌌 Methodology Validation: d=3 High-Density Statistical Sweep (PARALLEL)")
    print(f" -> Points to scan: {p_range}")
    print(f" -> Target Trials:  {num_trials}")
    print("==================================================================")

    # Use all but 2 cores, min 1
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Utilizing {num_cores} parallel workers...")

    for p in p_range:
        p_str = f"{p:.4f}"
        if p_str in results["points"]:
            print(f"⏭️ Skipping p = {p_str} (already completed).")
            continue
            
        print(f"\n--- Scanning Physical Base Rate: p = {p_str} ---")
        
        # Build task arguments
        task_args = [(p, seed, trial) for trial in range(num_trials)]
        
        failures_mwpm = 0
        lifetimes_logical = []

        with multiprocessing.Pool(num_cores) as pool:
            completed = 0
            for is_mwpm_fail, L_logical in pool.imap_unordered(run_single_trial, task_args):
                if is_mwpm_fail:
                    failures_mwpm += 1
                if L_logical is not None:
                    lifetimes_logical.append(L_logical)
                
                completed += 1
                if completed % 100 == 0:
                    print(f"    Progress: {completed}/{num_trials} trials completed...", flush=True)

        p_fail_mwpm = failures_mwpm / num_trials
        avg_l_logical = float(np.mean(lifetimes_logical)) if lifetimes_logical else 0.0
        std_l_logical = float(np.std(lifetimes_logical)) if lifetimes_logical else 0.0
        var_l_logical = float(np.var(lifetimes_logical)) if lifetimes_logical else 0.0
        
        results["points"][p_str] = {
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
        
        print(f" -> Metrics: P_fail(MWPM) = {p_fail_mwpm:.3f} | <L_logical> = {avg_l_logical:.3f} ± {std_l_logical:.3f}")

    print("\n==================================================================")
    print(f"✅ High-Density Sweep Finalized: {output_path}")
    print("==================================================================")
    return results

if __name__ == '__main__':
    dense_p_range = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    run_dense_d3_sweep(dense_p_range, num_trials=500)
