import numpy as np
import pytest
from src.geometry.spacetime import SpacetimeComplex
from src.enumeration.scaling_engine import ScalingDPEngine
from src.enumeration.kernel import SpatialKernelCache
from src.enumeration.dp_engine import ExactDPEngine

def test_scaling_engine_vacuum_agreement():
    """
    Verifies that the sparse ScalingDPEngine produces identical marginal probability assignments
    to the ExactDPEngine under vacuum syndrome configurations at d=3.
    """
    d = 3
    st_complex = SpacetimeComplex(d=d, T=d)
    s_obs = np.zeros((st_complex.T + 1, d, d), dtype=int)
    
    # 1. Exact base engine
    cache = SpatialKernelCache(d=d, p_spatial=0.01)
    exact_engine = ExactDPEngine(st_complex, kernel_cache=cache, p_temporal=0.01)
    exact_marginals = exact_engine.compute_edge_marginals(s_obs)
    
    # 2. Sparse scaling engine
    scaling_engine = ScalingDPEngine(st_complex, p_spatial=0.01, p_temporal=0.01, k_max=2, epsilon_cutoff=1e-12)
    scaling_marginals = scaling_engine.compute_edge_marginals(s_obs)
    
    # Assert keys match precisely
    assert set(exact_marginals.keys()) == set(scaling_marginals.keys())
    
    # Assert marginal probabilities align within extremely tight numerical tolerance
    for edge in exact_marginals:
        np.testing.assert_allclose(exact_marginals[edge], scaling_marginals[edge], atol=1e-5, rtol=1e-3)

def test_scaling_engine_d5_dryrun():
    """
    Verifies that the ScalingDPEngine successfully executes memory-bounded Forward-Backward
    tensor contractions on a d=5 lattice without memory overflow.
    """
    d = 5
    st_complex = SpacetimeComplex(d=d, T=d)
    s_obs = np.zeros((st_complex.T + 1, d, d), dtype=int)
    
    # Inject a clean sparse fault pair signature
    s_obs[1, 2, 2] = 1
    s_obs[2, 2, 2] = 1
    
    scaling_engine = ScalingDPEngine(st_complex, p_spatial=0.01, p_temporal=0.01, k_max=2, epsilon_cutoff=1e-10)
    marginals = scaling_engine.compute_edge_marginals(s_obs)
    
    assert len(marginals) > 0
    # Confirm maximum evaluated probability remains bounded
    assert max(marginals.values()) <= 1.0
    assert min(marginals.values()) >= 0.0
