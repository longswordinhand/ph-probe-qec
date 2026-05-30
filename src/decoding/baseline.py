import math
import numpy as np
import pymatching
from typing import Dict, Tuple, List, Any
from src.geometry.spacetime import SpacetimeComplex

class MWPMBaselineDecoder:
    r"""
    Implements standard Minimum Weight Perfect Matching (MWPM) decoding via PyMatching.
    
    Operates strictly under unconditioned classic classical assumptions to establish baseline fairness.
    Edge log-likelihood weights bypass dynamic sub-partition updates:
    w_0(e) = \ln\left(\frac{1 - p_e}{p_e}\right)
    
    Evaluates net logical failures by projecting combined physical + correction fault chains
    directly across homological Poincare dual boundary sections.
    """
    def __init__(self, st_complex: SpacetimeComplex, p_spatial: float = 0.01, p_temporal: float = 0.01):
        self.complex = st_complex
        self.p_spatial = p_spatial
        self.p_temporal = p_temporal
        
        self.num_vertices = self.complex.num_grid_vertices
        self.matching = pymatching.Matching()
        self.edge_map: List[Tuple[int, int]] = []
        
        self._build_matching_graph()

    def _canonical_edge(self, u: int, v: int) -> Tuple[int, int]:
        return tuple(sorted((u, v)))

    def _build_matching_graph(self):
        r"""
        Compiles the classical matching graph topology mapping cellular vertex identifiers.
        Assigns distinct fault tracking index vectors to evaluate macroscopic logical loop flips.
        """
        for edge_idx, edge_data in enumerate(self.complex.all_edges()):
            u, v = edge_data['vertices']
            c_edge = self._canonical_edge(u, v)
            self.edge_map.append(c_edge)
            
            p = self.p_spatial if edge_data['type'].startswith('spatial') else self.p_temporal
            # Prevent log(0) numeric singularities
            p = min(0.9999, max(0.0001, p))
            weight = math.log((1.0 - p) / p)
            
            # Add weighted connection directly mapping complex vertex identifiers
            self.matching.add_edge(u, v, weight=weight, fault_ids=edge_idx)

    def decode_syndrome(self, s_obs: np.ndarray, actual_faults: Dict[Tuple[int, int], int]) -> Tuple[bool, bool]:
        r"""
        Resolves optimal matching paths from observed binary syndrome sets.
        
        Evaluates homological failures: Returns (failure_x, failure_y) booleans indicating
        whether combined net physical errors + decoder corrections flipped macroscopic logical invariants.
        """
        # Pack localized 3D syndrome tensors into absolute flat 1D arrays matching graph layout
        syndrome_array = np.zeros(self.num_vertices, dtype=np.uint8)
        for t in range(self.complex.T + 1):
            for y in range(self.complex.d):
                for x in range(self.complex.d):
                    v_id = self.complex.coord_to_id(x, y, t)
                    syndrome_array[v_id] = s_obs[t, y, x]

        # Resolve MWPM correction path bit vector
        correction_vector = self.matching.decode(syndrome_array)
        
        # Merge classical correction predictions with real physical configurations
        total_faults = np.zeros(len(self.edge_map), dtype=int)
        for idx, c_edge in enumerate(self.edge_map):
            phys_val = actual_faults.get(c_edge, 0)
            total_faults[idx] = phys_val ^ int(correction_vector[idx])

        # Evaluate net topological logical sector failures via Poincare dual transversal cross-sections
        # Horizontal logical loops cross the vertical dual plane (e.g. crossing x=0 spatial edges)
        wind_x_flips = 0
        # Vertical logical loops cross the horizontal dual plane (e.g. crossing y=0 spatial edges)
        wind_y_flips = 0
        
        for idx, edge_data in enumerate(self.complex.all_edges()):
            if total_faults[idx] == 1:
                # To assess macroscopic logical boundaries consistently, we project multi-slice paths onto base cross-sections
                etype = edge_data['type']
                x, y = edge_data['x'], edge_data['y']
                
                if etype == 'spatial_h' and x == 0:
                    wind_x_flips += 1
                elif etype == 'spatial_v' and y == 0:
                    wind_y_flips += 1

        failure_x = (wind_x_flips % 2) != 0
        failure_y = (wind_y_flips % 2) != 0
        
        return failure_x, failure_y
