import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from src.geometry.spacetime import SpacetimeComplex

class NoiseModel:
    r"""
    Generates stochastic physical noise and extracts precise topological syndrome boundaries.
    
    Physical noise models track independent fault mechanisms:
    - Spatial edges correspond to data qubit errors (X or Z chain components).
    - Temporal edges correspond to syndrome measurement fault trajectories.
    
    The cellular boundary operator maps 1-chains (errors) to 0-chains (syndromes):
    \partial E = S_{\mathrm{obs}}
    """
    def __init__(self, st_complex: SpacetimeComplex, p_spatial: float = 0.01, p_temporal: float = 0.01):
        self.complex = st_complex
        self.p_spatial = p_spatial
        self.p_temporal = p_temporal

    def _canonical_edge(self, u: int, v: int) -> Tuple[int, int]:
        return tuple(sorted((u, v)))

    def sample_error_configuration(self, seed: Optional[int] = None) -> Dict[Tuple[int, int], int]:
        r"""
        Samples independent physical faults across the entire cellular lattice.
        Returns a canonical mapping of active fault assignments.
        """
        if seed is not None:
            np.random.seed(seed)
            
        error_config = {}
        for edge_data in self.complex.all_edges():
            u, v = edge_data['vertices']
            c_edge = self._canonical_edge(u, v)
            
            p = self.p_spatial if edge_data['type'].startswith('spatial') else self.p_temporal
            err_val = 1 if np.random.rand() < p else 0
            error_config[c_edge] = err_val
            
        return error_config

    def compute_syndrome(self, error_config: Dict[Tuple[int, int], int]) -> np.ndarray:
        r"""
        Extracts the exact cellular boundary syndrome map S_{\mathrm{obs}}(t, y, x).
        
        By fundamental homological boundary definitions, the observed syndrome at any measurement event
        vertex is the mod-2 parity sum of all incident physical edges hosting an active fault.
        """
        s_obs = np.zeros((self.complex.T + 1, self.complex.d, self.complex.d), dtype=int)
        
        for edge_data in self.complex.all_edges():
            u, v = edge_data['vertices']
            c_edge = self._canonical_edge(u, v)
            err_val = error_config.get(c_edge, 0)
            
            if err_val == 1:
                # Active fault flips the measurement outcome at both endpoint bounding vertices
                ux, uy, ut = self.complex.id_to_coord(u)
                vx, vy, vt = self.complex.id_to_coord(v)
                
                s_obs[ut, uy, ux] ^= 1
                s_obs[vt, vy, vx] ^= 1
                
        # Topological Invariant Check: Since every 1-cell bounds precisely two 0-cells,
        # the net global parity sum across the entire compact spatiotemporal boundary must evaluate to zero mod 2.
        assert np.sum(s_obs) % 2 == 0, "Cellular boundary output violated compact even-parity invariants."
        return s_obs

    def generate_custom_fault(self, active_edges: List[Tuple[str, int, int, int]]) -> Dict[Tuple[int, int], int]:
        r"""
        Utility to manually instantiate deterministic fault configurations.
        Accepts targeted list of edge descriptors: (type, t, x, y).
        """
        lookup = {}
        for edge_data in self.complex.all_edges():
            key = (edge_data['type'], edge_data['t'], edge_data['x'], edge_data['y'])
            u, v = edge_data['vertices']
            lookup[key] = self._canonical_edge(u, v)
            
        error_config = {}
        for key in active_edges:
            if key not in lookup:
                raise KeyError(f"Targeted edge descriptor {key} missing from complex layout.")
            error_config[lookup[key]] = 1
            
        return error_config
