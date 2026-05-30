import os
import json
import math
import numpy as np
from typing import Dict, List, Any
from src.geometry.spacetime import SpacetimeComplex
from src.syndrome.noise import NoiseModel
from src.enumeration.kernel import SpatialKernelCache
from src.enumeration.dp_engine import ExactDPEngine
from src.persistence.homology import HomologyEngine
from src.decoding.baseline import MWPMBaselineDecoder

def run_physical_parameter_sweep(p_range: List[float], num_trials: int = 10, output_path: str = "results/phase1_sweep_data.json"):
    r"""
    Orchestrates systemic parameter sweeps driving parallel execution streams across physical error rates.
    
    Integrates our complete verification triad:
    Physical Sample -> Exact DP Kernel Sub-Partitions -> Homology Extraction vs PyMatching MWPM Baseline.
    
    Exports aggregated logical lifetime averages and decoding failure probabilities to persistent JSON.
    """
    results = {}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Initiating Parameter Sweep across {len(p_range)} error profiles ({num_trials} Monte Carlo trials per scale)...")

    for p in p_range:
        print(f"\n--- Scanning Physical Base Rate: p = {p:.4f} ---")
        complex_lattice = SpacetimeComplex(d=3, T=2)
        
        # Instantiate execution modules mapped to local error parameters
        noise = NoiseModel(complex_lattice, p_spatial=p, p_temporal=p)
        kernel = SpatialKernelCache(d=3, p_spatial=p)
        dp_engine = ExactDPEngine(complex_lattice, kernel, p_temporal=p)
        mwpm = MWPMBaselineDecoder(complex_lattice, p_spatial=p, p_temporal=p)
        
        failures_mwpm = 0
        lifetimes_logical = []

        for trial in range(num_trials):
            # 1. Sample native cellular physical faults
            faults = noise.sample_error_configuration()
            s_obs = noise.compute_syndrome(faults)
            
            # 2. Evaluate conventional MWPM logical failure statistics
            fail_x, fail_y = mwpm.decode_syndrome(s_obs, faults)
            if fail_x or fail_y:
                failures_mwpm += 1
                
            # 3. Propagate exact Forward-Backward DP sub-partition filters
            try:
                marginals = dp_engine.compute_edge_marginals(s_obs)
            except ValueError:
                # Handle vacuum zero-cycle boundary anomalies
                continue
                
            # 4. Map Bayesian marginals to filtration weights
            export_data = {'simplices': [], 'max_weight': -math.log(1e-5)}
            cap_weight = export_data['max_weight']
            
            for edge_data in complex_lattice.all_edges():
                u, v = edge_data['vertices']
                c_edge = tuple(sorted((u, v)))
                prob = marginals.get(c_edge, 1e-6)
                prob = min(0.9999, max(1e-6, prob))
                w_e = -math.log(prob)
                
                export_data['simplices'].append({
                    'vertices': [u, v],
                    'weight': min(cap_weight, w_e),
                    'type': edge_data['type']
                })
                
            # 5. Extract multi-boundary persistence lengths via GUDHI solvers
            homology = HomologyEngine(complex_lattice, export_data)
            obs = homology.compute_observables(persistence_threshold=0.5, cap_infinity=cap_weight)
            lifetimes_logical.append(obs['L_logical'])

        p_fail_mwpm = failures_mwpm / num_trials
        avg_l_logical = float(np.mean(lifetimes_logical)) if lifetimes_logical else 0.0
        
        results[f"{p:.4f}"] = {
            'p_physical': p,
            'p_fail_mwpm': p_fail_mwpm,
            'avg_L_logical': avg_l_logical,
            'trials_evaluated': len(lifetimes_logical)
        }
        
        print(f" -> Averaged Deliverables: P_fail(MWPM) = {p_fail_mwpm:.3f} | <L_logical> = {avg_l_logical:.3f}")

    # Export canonical deliverables
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSweep successfully finalized. Structured metrics exported to: {output_path}")
    return results

if __name__ == '__main__':
    # Default execution config optimized for quick verification scaling
    test_scales = [0.002, 0.01, 0.05, 0.12]
    run_physical_parameter_sweep(test_scales, num_trials=5)
