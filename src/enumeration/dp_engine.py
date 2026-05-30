import numpy as np
from typing import Dict, Tuple, List, Any
from src.geometry.spacetime import SpacetimeComplex
from src.enumeration.kernel import SpatialKernelCache

class ExactDPEngine:
    r"""
    Executes the highly optimized Forward-Backward Dynamic Programming enumeration tensor contractions.
    
    State composite representation:
    s_{\mathrm{id}} = e_{\mathrm{time\_mask}} | (w_x \ll 9) | (w_y \ll 10)
    Where e_{\mathrm{time\_mask}} captures pending multi-qubit temporal fault trajectories (512 states),
    and (w_x, w_y) captures global macroscopic logical sector windings (4 states).
    Total state space dimensionality per slice: 2048 discrete states.
    
    Outputs absolute Bayesian single-edge marginal error probability maps parameterized for filtration.
    """
    def __init__(self, st_complex: SpacetimeComplex, kernel_cache: SpatialKernelCache, p_temporal: float = 0.01):
        self.complex = st_complex
        self.kernel_cache = kernel_cache
        self.p_temporal = p_temporal
        
        self.num_v_slice = self.complex.d * self.complex.d
        self.num_t_states = 1 << self.num_v_slice
        self.total_states = self.num_t_states * 4
        
        # Pre-cache temporal cluster fault assignment weights
        self._w_time = np.array([
            (p_temporal ** k) * ((1.0 - p_temporal) ** (self.num_v_slice - k))
            for k in range(self.num_v_slice + 1)
        ])
        self.w_time_arr = np.array([
            self._w_time[mask.bit_count()] for mask in range(self.num_t_states)
        ])

        # Pre-extract valid spatial boundary kernels (even parity optimization)
        # Spatial boundaries evaluate to zero global parity invariant
        self.valid_b_sp = np.array([b for b in range(self.num_t_states) if b.bit_count() % 2 == 0], dtype=int)

    def _state_id(self, e_mask: int, wx: int, wy: int) -> int:
        return e_mask | (wx << 9) | (wy << 10)

    def _decode_state(self, s_id: int) -> Tuple[int, int, int]:
        e_mask = s_id & 511
        wx = (s_id >> 9) & 1
        wy = (s_id >> 10) & 1
        return (e_mask, wx, wy)

    def _syndrome_slice_mask(self, s_obs: np.ndarray, t: int) -> int:
        r"""Packs localized 2D syndrome matrices into highly dense integer bit masks."""
        mask = 0
        for y in range(self.complex.d):
            for x in range(self.complex.d):
                if s_obs[t, y, x]:
                    mask |= (1 << (y * self.complex.d + x))
        return mask

    def execute_forward_backward(self, s_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        r"""
        Propagates Bayesian statistical partition functions across the cellular complex interfaces.
        
        Forward pass (\alpha_t): Accumulates conditioned incoming partition sums.
        Backward pass (\beta_t): Propagates terminal compatibility matching criteria.
        """
        T = self.complex.T
        alpha = np.zeros((T + 1, self.total_states), dtype=float)
        beta = np.zeros((T + 1, self.total_states), dtype=float)

        # Pre-pack observed temporal layer syndrome masks
        s_masks = [self._syndrome_slice_mask(s_obs, t) for t in range(T + 1)]

        # --- 1. FORWARD PASS ---
        # Base initialization prior to dynamic injection slice t=0
        # Vacuum interface state assumes absolute zero initial error history
        e_prev_base = 0
        w_prev_x, w_prev_y = 0, 0
        
        # Advance into temporal boundary slice t=0
        s_target = s_masks[0]
        for b_sp in self.valid_b_sp:
            for wx in range(2):
                for wy in range(2):
                    w_kernel = self.kernel_cache.kernel[b_sp, wx, wy]
                    if w_kernel == 0:
                        continue
                        
                    e_next = s_target ^ b_sp ^ e_prev_base
                    w_next_x = w_prev_x ^ wx
                    w_next_y = w_prev_y ^ wy
                    
                    s_next = self._state_id(e_next, w_next_x, w_next_y)
                    # Accumulate joint transition path tensor metrics
                    alpha[0, s_next] += w_kernel * self.w_time_arr[e_next]

        # Sequential Forward Propagation
        for t in range(1, T + 1):
            s_target = s_masks[t]
            # To maximize raw computational throughput, iterate purely over active incoming states
            active_states = np.nonzero(alpha[t - 1])[0]
            
            for s_prev in active_states:
                a_prev = alpha[t - 1, s_prev]
                e_prev, w_prev_x, w_prev_y = self._decode_state(s_prev)
                
                # Vectorized over valid spatial boundary syndromes
                e_next = s_target ^ self.valid_b_sp ^ e_prev
                w_t_factor = self.w_time_arr[e_next]
                
                for wx in range(2):
                    w_next_x = w_prev_x ^ wx
                    for wy in range(2):
                        w_next_y = w_prev_y ^ wy
                        w_kernel = self.kernel_cache.kernel[self.valid_b_sp, wx, wy]
                        
                        mask = w_kernel > 0
                        if not np.any(mask):
                            continue
                            
                        s_next = e_next[mask] | (w_next_x << 9) | (w_next_y << 10)
                        alpha[t, s_next] += a_prev * w_kernel[mask] * w_t_factor[mask]

        # Total constrained partition function evaluates over valid closed boundary cycles.
        # Terminal temporal slice T forces an absolute boundary constraint: terminal virtual temporal faults must vanish (e_next \equiv 0).
        z_total = sum(alpha[T, self._state_id(0, wx, wy)] for wx in range(2) for wy in range(2))
        if z_total == 0:
            raise ValueError("Observed syndrome history yields absolutely zero valid physical homology chains.")

        # --- 2. BACKWARD PASS ---
        # Initialize terminal interface boundaries enforcing absolute homological closure constraints
        for wx in range(2):
            for wy in range(2):
                beta[T, self._state_id(0, wx, wy)] = 1.0

        # Sequential Backward Propagation
        for t in range(T, 0, -1):
            s_target = s_masks[t]
            active_next_states = np.nonzero(beta[t])[0]
            
            # To compute backward transitions effectively, we resolve incoming states compatible with outgoing targets
            for s_next in active_next_states:
                b_next = beta[t, s_next]
                e_next, w_next_x, w_next_y = self._decode_state(s_next)
                w_t_factor = self.w_time_arr[e_next]
                
                # Vectorized over valid spatial boundary syndromes
                e_prev = s_target ^ self.valid_b_sp ^ e_next
                
                for wx in range(2):
                    w_prev_x = w_next_x ^ wx
                    for wy in range(2):
                        w_prev_y = w_next_y ^ wy
                        w_kernel = self.kernel_cache.kernel[self.valid_b_sp, wx, wy]
                        
                        mask = w_kernel > 0
                        if not np.any(mask):
                            continue
                            
                        s_prev = e_prev[mask] | (w_prev_x << 9) | (w_prev_y << 10)
                        beta[t - 1, s_prev] += w_kernel[mask] * w_t_factor * b_next

        return alpha, beta, z_total

    def compute_edge_marginals(self, s_obs: np.ndarray) -> Dict[Tuple[int, int], float]:
        r"""
        Extracts Bayesian edge marginal probability matrices \mu(e) mapping precise physical fault scales.
        
        Leverages full cached tensor derivative evaluations to solve constrained sub-partition sums natively.
        """
        alpha, beta, z_total = self.execute_forward_backward(s_obs)
        T = self.complex.T
        marginals = {}
        s_masks = [self._syndrome_slice_mask(s_obs, t) for t in range(T + 1)]

        # 1. Temporal Edge Marginals
        # Temporal error edge assignments are stored directly inside outgoing state ID masks
        for t in range(T):
            # The physical state of temporal edges bridging layer t to t+1 is captured by alpha[t] output masks
            for y in range(self.complex.d):
                for x in range(self.complex.d):
                    u = self.complex.coord_to_id(x, y, t)
                    v = self.complex.coord_to_id(x, y, t + 1)
                    c_edge = tuple(sorted((u, v)))
                    
                    bit_offset = y * self.complex.d + x
                    mask_check = 1 << bit_offset
                    
                    # Sum joint conditioned probabilities over compatible active states
                    prob_sum = 0.0
                    for s_id in range(self.total_states):
                        if (s_id & 511) & mask_check:
                            prob_sum += alpha[t, s_id] * beta[t, s_id]
                            
                    marginals[c_edge] = min(1.0, max(0.0, prob_sum / z_total))

        # 2. Spatial Edge Marginals
        # Evaluated via precise inner loop tensor dot products integrating restricted single-edge kernel matrices
        for t in range(T + 1):
            s_target = s_masks[t]
            
            # Extract active states to accelerate restricted matrix inner dot products
            if t == 0:
                e_prev = np.array([0])
                w_prev_x = np.array([0])
                w_prev_y = np.array([0])
                a_prev = np.array([1.0])
            else:
                active_indices = np.nonzero(alpha[t - 1])[0]
                a_prev = alpha[t - 1, active_indices]
                e_prev, w_prev_x, w_prev_y = self._decode_state(active_indices)
                
            e_next = s_target ^ self.valid_b_sp[:, None] ^ e_prev[None, :]
            w_t_factor = self.w_time_arr[e_next]
            
            for edge_idx, (etype, x, y) in enumerate(self.kernel_cache.edge_ordering):
                u = self.complex.coord_to_id(x, y, t)
                if etype == 'spatial_h':
                    v = self.complex.coord_to_id((x + 1) % self.complex.d, y, t)
                else:
                    v = self.complex.coord_to_id(x, (y + 1) % self.complex.d, t)
                c_edge = tuple(sorted((u, v)))
                
                restricted_kernel = self.kernel_cache.restricted_kernels[edge_idx]
                prob_sum = 0.0
                
                for wx in range(2):
                    w_next_x = w_prev_x ^ wx
                    for wy in range(2):
                        w_next_y = w_prev_y ^ wy
                        w_k_rest = restricted_kernel[self.valid_b_sp, wx, wy]
                        
                        mask_b = w_k_rest > 0
                        if not np.any(mask_b):
                            continue
                            
                        w_k_sliced = w_k_rest[mask_b]
                        e_next_sliced = e_next[mask_b, :]
                        w_t_factor_sliced = w_t_factor[mask_b, :]
                        
                        s_next = e_next_sliced | (w_next_x[None, :] << 9) | (w_next_y[None, :] << 10)
                        term = (a_prev[None, :] * w_k_sliced[:, None]) * w_t_factor_sliced * beta[t, s_next]
                        prob_sum += np.sum(term)
                                
                marginals[c_edge] = min(1.0, max(0.0, prob_sum / z_total))

        return marginals
