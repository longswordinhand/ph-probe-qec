import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import Dict, Tuple, List, Any
from src.geometry.spacetime import SpacetimeComplex

class ScalingDPEngine:
    r"""
    Executes the Forward-Backward Dynamic Programming enumeration kernel optimized for arbitrary lattice scales (d >= 5).
    
    Bypasses exponential O(2^{2d^2}) static transfer matrix allocations by leveraging:
    1. Sparse dictionary state transitions (Dynamic State Pruning via \epsilon_{\mathrm{cutoff}}).
    2. Low-weight spatial layer error expansion (enumerating single-slice spatial errors up to k_{\mathrm{max}}).
    
    State mapping utilizes absolute sparse dictionary structures keyed by:
    s_{\mathrm{key}} = (e_{\mathrm{time\_mask}}, w_x, w_y)
    Where e_{\mathrm{time\_mask}} is the standard integer bitmask representing active temporal edge assignments bridging slice t to t+1.
    """
    def __init__(self, st_complex: SpacetimeComplex, p_spatial: float = 0.01, p_temporal: float = 0.01, 
                 k_max: int = 2, epsilon_cutoff: float = 1e-12):
        self.complex = st_complex
        self.d = self.complex.d
        self.num_v_slice = self.d * self.d
        self.num_sp_edges = 2 * self.num_v_slice
        
        self.p_spatial = p_spatial
        self.p_temporal = p_temporal
        self.k_max = k_max
        self.epsilon_cutoff = epsilon_cutoff
        
        # Pre-cache single slice spatial configuration expansions
        # Store as list of tuples: (b_sp_mask, wx, wy, weight, active_edge_indices)
        self.sparse_sp_configs: List[Tuple[int, int, int, float, Tuple[int, ...]]] = []
        self._edge_ordering: List[Tuple[str, int, int]] = []
        self._edge_masks: List[int] = []
        self._wind_x_mask = 0
        self._wind_y_mask = 0
        
        self._build_spatial_expansion()

    def _build_spatial_expansion(self):
        r"""Pre-builds canonical sparse expansion configurations up to k_{\mathrm{max}} spatial faults."""
        # 1. Edge ordering mapping
        for y in range(self.d):
            for x in range(self.d):
                self._edge_ordering.append(('spatial_h', x, y))
        for y in range(self.d):
            for x in range(self.d):
                self._edge_ordering.append(('spatial_v', x, y))
                
        # 2. Boundary inclusion masks
        for idx, (etype, x, y) in enumerate(self._edge_ordering):
            u_idx = y * self.d + x
            if etype == 'spatial_h':
                v_idx = y * self.d + ((x + 1) % self.d)
            else:
                v_idx = ((y + 1) % self.d) * self.d + x
            self._edge_masks.append((1 << u_idx) | (1 << v_idx))
            
        # 3. Winding validation cross-sections
        for idx, (et, x, y) in enumerate(self._edge_ordering):
            if et == 'spatial_h' and x == 0:
                self._wind_x_mask |= (1 << idx)
            if et == 'spatial_v' and y == 0:
                self._wind_y_mask |= (1 << idx)
                
        # Pre-cache temporal and spatial base decay rates
        w_base_sp = (1.0 - self.p_spatial) ** self.num_sp_edges
        factor_sp = self.p_spatial / (1.0 - self.p_spatial) if self.p_spatial < 1.0 else 0.0
        
        # Enumerate low-weight configurations: k \in [0, k_max]
        # Only even-parity boundary combinations survive global constraint validity
        for k in range(self.k_max + 1):
            w_k = w_base_sp * (factor_sp ** k)
            for active_indices in combinations(range(self.num_sp_edges), k):
                b_sp = 0
                cfg_mask = 0
                for idx in active_indices:
                    b_sp ^= self._edge_masks[idx]
                    cfg_mask |= (1 << idx)
                    
                # Strict optimization: global state validity mandates even-parity localized syndrome excitations
                if b_sp.bit_count() % 2 != 0:
                    continue
                    
                wx = (cfg_mask & self._wind_x_mask).bit_count() & 1
                wy = (cfg_mask & self._wind_y_mask).bit_count() & 1
                
                self.sparse_sp_configs.append((b_sp, wx, wy, w_k, active_indices))

    def _syndrome_mask(self, s_obs: np.ndarray, t: int) -> int:
        mask = 0
        for y in range(self.d):
            for x in range(self.d):
                if s_obs[t, y, x]:
                    mask |= (1 << (y * self.d + x))
        return mask

    def _t_weight(self, e_mask: int) -> float:
        k = e_mask.bit_count()
        return (self.p_temporal ** k) * ((1.0 - self.p_temporal) ** (self.num_v_slice - k))

    def _prune_distribution(self, dist: Dict[Tuple[int, int, int], float]) -> Dict[Tuple[int, int, int], float]:
        r"""Executes dynamic state truncation to enforce linear memory footprint containment."""
        if not dist:
            return dist
        max_p = max(dist.values())
        threshold = max_p * self.epsilon_cutoff
        return {k: v for k, v in dist.items() if v >= threshold}

    def execute_forward_backward(self, s_obs: np.ndarray) -> Tuple[List[Dict[Tuple[int, int, int], float]], 
                                                                   List[Dict[Tuple[int, int, int], float]], float]:
        T = self.complex.T
        s_masks = [self._syndrome_mask(s_obs, t) for t in range(T + 1)]
        
        # alpha[t] mapping keys: (e_next, wx, wy) -> joint forward probability sum
        alpha: List[Dict[Tuple[int, int, int], float]] = [defaultdict(float) for _ in range(T + 1)]
        beta: List[Dict[Tuple[int, int, int], float]] = [defaultdict(float) for _ in range(T + 1)]
        
        # --- 1. FORWARD PASS ---
        # Initialization slice t=0
        s_target = s_masks[0]
        for b_sp, wx, wy, w_sp, _ in self.sparse_sp_configs:
            e_next = s_target ^ b_sp
            w_t = self._t_weight(e_next)
            alpha[0][(e_next, wx, wy)] += w_sp * w_t
            
        alpha[0] = self._prune_distribution(alpha[0])
        
        # Sequential multi-layer propagation
        for t in range(1, T + 1):
            s_target = s_masks[t]
            current_alpha = alpha[t - 1]
            next_alpha = alpha[t]
            
            for (e_prev, w_prev_x, w_prev_y), a_prev in current_alpha.items():
                for b_sp, wx, wy, w_sp, _ in self.sparse_sp_configs:
                    e_next = s_target ^ b_sp ^ e_prev
                    w_t = self._t_weight(e_next)
                    
                    w_next_x = w_prev_x ^ wx
                    w_next_y = w_prev_y ^ wy
                    
                    next_alpha[(e_next, w_next_x, w_next_y)] += a_prev * w_sp * w_t
                    
            alpha[t] = self._prune_distribution(next_alpha)
            
        # Total partition summation over valid closed physical homology boundary chains
        z_total = sum(v for (e_m, wx, wy), v in alpha[T].items() if e_m == 0)
        if z_total == 0:
            raise ValueError("Forward propagation yielded zero valid physical boundary closure paths.")

        # --- 2. BACKWARD PASS ---
        # Terminal interface initialization enforces e_next == 0 closures
        for wx in range(2):
            for wy in range(2):
                if (0, wx, wy) in alpha[T]:
                    beta[T][(0, wx, wy)] = 1.0
                    
        # Sequential Backward trace
        for t in range(T, 0, -1):
            s_target = s_masks[t]
            current_beta = beta[t]
            prev_beta = beta[t - 1]
            
            for (e_next, w_next_x, w_next_y), b_next in current_beta.items():
                w_t = self._t_weight(e_next)
                for b_sp, wx, wy, w_sp, _ in self.sparse_sp_configs:
                    e_prev = s_target ^ b_sp ^ e_next
                    
                    w_prev_x = w_next_x ^ wx
                    w_prev_y = w_next_y ^ wy
                    
                    # Accumulate backward compatibilities conditioned on incoming forward active branches
                    if (e_prev, w_prev_x, w_prev_y) in alpha[t - 1]:
                        prev_beta[(e_prev, w_prev_x, w_prev_y)] += w_sp * w_t * b_next
                        
            beta[t - 1] = self._prune_distribution(prev_beta)
            
        return alpha, beta, z_total

    def compute_edge_marginals(self, s_obs: np.ndarray) -> Dict[Tuple[int, int], float]:
        alpha, beta, z_total = self.execute_forward_backward(s_obs)
        T = self.complex.T
        marginals = {}
        s_masks = [self._syndrome_mask(s_obs, t) for t in range(T + 1)]
        
        # 1. Temporal Edge Marginals
        for t in range(T):
            active_states = alpha[t]
            active_betas = beta[t]
            
            for y in range(self.d):
                for x in range(self.d):
                    u = self.complex.coord_to_id(x, y, t)
                    v = self.complex.coord_to_id(x, y, t + 1)
                    c_edge = tuple(sorted((u, v)))
                    
                    mask_check = 1 << (y * self.d + x)
                    prob_sum = sum(
                        a_val * active_betas.get(s_key, 0.0) 
                        for s_key, a_val in active_states.items() 
                        if s_key[0] & mask_check
                    )
                    marginals[c_edge] = min(1.0, max(0.0, prob_sum / z_total))
                    
        # 2. Spatial Edge Marginals
        # Pre-accumulate single-edge sparse weights to solve localized partition integrals efficiently
        for t in range(T + 1):
            s_target = s_masks[t]
            if t == 0:
                prev_states = [((0, 0, 0), 1.0)]
            else:
                prev_states = list(alpha[t - 1].items())
                
            active_betas = beta[t]
            
            # Initialize array to accumulate probabilities per spatial edge
            sp_prob_sums = np.zeros(self.num_sp_edges, dtype=float)
            
            for (e_prev, w_prev_x, w_prev_y), a_prev in prev_states:
                for b_sp, wx, wy, w_sp, active_indices in self.sparse_sp_configs:
                    if not active_indices:
                        continue
                        
                    e_next = s_target ^ b_sp ^ e_prev
                    w_next_x = w_prev_x ^ wx
                    w_next_y = w_prev_y ^ wy
                    s_next = (e_next, w_next_x, w_next_y)
                    
                    if s_next in active_betas:
                        path_weight = a_prev * w_sp * self._t_weight(e_next) * active_betas[s_next]
                        for idx in active_indices:
                            sp_prob_sums[idx] += path_weight
                            
            # Map canonical indexing to spatial coordinate edges
            for idx, (etype, x, y) in enumerate(self._edge_ordering):
                u = self.complex.coord_to_id(x, y, t)
                if etype == 'spatial_h':
                    v = self.complex.coord_to_id((x + 1) % self.d, y, t)
                else:
                    v = self.complex.coord_to_id(x, (y + 1) % self.d, t)
                c_edge = tuple(sorted((u, v)))
                
                marginals[c_edge] = min(1.0, max(0.0, sp_prob_sums[idx] / z_total))
                
        return marginals
