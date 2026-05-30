import numpy as np
from typing import Dict, Tuple, List, Any

class SpatialKernelCache:
    r"""
    Precomputes immutable local spatial layer transfer kernels for exact dynamic programming.
    
    For a single spatial slice on a d=3 lattice, there are 2d^2 = 18 physical spatial edges.
    Exhaustively enumerating all 2^{18} = 262,144 spatial configurations enables absolute
    pre-aggregation of statistical partition weights parameterized by:
    1. Emerging local syndrome boundary mask B_{\mathrm{sp}} (2^9 states, constrained to even parity).
    2. Homological logical winding sector (w_x, w_y) \in \{0, 1\}^2.
    
    This technique mathematically compresses run-time inner tensor iterations from O(2^{18}) down to O(1).
    """
    def __init__(self, d: int = 3, p_spatial: float = 0.01):
        self.d = d
        self.num_vertices = d * d
        self.num_edges = 2 * self.num_vertices
        self.p_spatial = p_spatial
        
        # Pre-calculated statistical weights indexed by Hamming weight (error counts)
        self._weights_by_k = np.array([
            (p_spatial ** k) * ((1.0 - p_spatial) ** (self.num_edges - k))
            for k in range(self.num_edges + 1)
        ])

        # Primary spatial transfer kernel: W_sp[b_sp_mask, wind_x, wind_y]
        self.kernel = np.zeros((1 << self.num_vertices, 2, 2), dtype=float)
        
        # Restricted kernels for exact single-edge Backward derivative evaluations
        # restricted_kernels[edge_idx, b_sp_mask, wind_x, wind_y]
        self.restricted_kernels = np.zeros((self.num_edges, 1 << self.num_vertices, 2, 2), dtype=float)
        
        # Maps geometric edge sequence to specific canonical index
        self.edge_ordering: List[Tuple[str, int, int]] = []
        self._build_ordering()
        self._precompute()

    def _build_ordering(self):
        r"""
        Establishes fixed canonical bit alignment mapping index -> spatial edge descriptor.
        First d^2 bits map to horizontal edges; remaining d^2 map to vertical edges.
        """
        for y in range(self.d):
            for x in range(self.d):
                self.edge_ordering.append(('spatial_h', x, y))
        for y in range(self.d):
            for x in range(self.d):
                self.edge_ordering.append(('spatial_v', x, y))

    def _precompute(self):
        r"""
        Executes the exhaustive state space iteration over all 2^{18} spatial fault configurations.
        Aggregates statistical cluster weights according to local homological boundary conditions.
        """
        # Pre-build local vertex boundary inclusion masks for bitwise speedups
        # Each edge endpoints trigger XOR bit flips in the output syndrome mask
        edge_masks = np.zeros(self.num_edges, dtype=int)
        for idx, (etype, x, y) in enumerate(self.edge_ordering):
            u_idx = y * self.d + x
            if etype == 'spatial_h':
                v_idx = y * self.d + ((x + 1) % self.d)
            else:
                v_idx = ((y + 1) % self.d) * self.d + x
            edge_masks[idx] = (1 << u_idx) | (1 << v_idx)

        # Indices of edges crossing the canonical Poincare dual cross-sections
        # Horizontal edges crossing x=0 cross-section evaluate logical wind_x parity
        wind_x_mask = sum(1 << idx for idx, (et, x, y) in enumerate(self.edge_ordering) if et == 'spatial_h' and x == 0)
        # Vertical edges crossing y=0 cross-section evaluate logical wind_y parity
        wind_y_mask = sum(1 << idx for idx, (et, x, y) in enumerate(self.edge_ordering) if et == 'spatial_v' and y == 0)

        # Iterate over all possible integer state configs: 0 to 262,143
        total_configs = 1 << self.num_edges
        
        # Utilize fast array ops for vector iteration speedups
        for cfg in range(total_configs):
            # Evaluate Hamming weight
            k = cfg.bit_count()
            weight = self._weights_by_k[k]

            # Compute resulting syndrome mask by XORing active edge boundary masks
            # For d=3, we can rapidly extract active bits
            b_sp_mask = 0
            temp = cfg
            idx = 0
            while temp > 0:
                if temp & 1:
                    b_sp_mask ^= edge_masks[idx]
                temp >>= 1
                idx += 1

            # Extract macro logical homological winding parities via Poincare section intersection parity
            wx = (cfg & wind_x_mask).bit_count() & 1
            wy = (cfg & wind_y_mask).bit_count() & 1

            # Populate primary aggregated tensor kernel
            self.kernel[b_sp_mask, wx, wy] += weight

            # Populate single-edge restricted kernels
            temp = cfg
            idx = 0
            while temp > 0:
                if temp & 1:
                    self.restricted_kernels[idx, b_sp_mask, wx, wy] += weight
                temp >>= 1
                idx += 1

    def get_edge_index(self, edge_type: str, x: int, y: int) -> int:
        r"""Retrieves the canonical bit integer offset for a specific grid edge."""
        base = 0 if edge_type == 'spatial_h' else self.num_vertices
        return base + y * self.d + x
