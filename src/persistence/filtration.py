import math
from typing import Dict, Tuple, List, Optional, Any
from src.geometry.spacetime import SpacetimeComplex

class FiltrationEngine:
    r"""
    Manages the physical mapping of error probabilities to TDA filtration weights.
    
    Filtration mapping rule:
    w(e) = -\log \mu(e)
    Where \mu(e) is the marginal error probability of edge e.
    
    High probability errors (\mu \to 1) yield early filtration weights (w \to 0).
    Low probability background noise (\mu \to 0) yields late weights (w \gg 1).
    """
    def __init__(self, st_complex: SpacetimeComplex, default_bg_prob: float = 1e-4):
        self.complex = st_complex
        self.default_bg_prob = default_bg_prob
        self.default_bg_weight = -math.log(default_bg_prob)
        # Maps canonical sorted vertex pairs to their computed filtration weight
        self.edge_weights: Dict[Tuple[int, int], float] = {}
        self._edge_lookup: Dict[Tuple[str, int, int, int], Tuple[int, int]] = {}
        
        self._initialize_weights()

    def _canonical_edge(self, u: int, v: int) -> Tuple[int, int]:
        return tuple(sorted((u, v)))

    def _initialize_weights(self):
        """Initializes all lattice edges with the default background noise weight."""
        for edge_data in self.complex.all_edges():
            u, v = edge_data['vertices']
            c_edge = self._canonical_edge(u, v)
            self.edge_weights[c_edge] = self.default_bg_weight
            
            # Populate descriptive lookup table for precise custom targeting
            key = (edge_data['type'], edge_data['t'], edge_data['x'], edge_data['y'])
            self._edge_lookup[key] = c_edge

    def set_edge_probability(self, edge_type: str, t: int, x: int, y: int, prob: float):
        """
        Manually injects an active error probability onto a specific grid edge.
        Useful for setting up distinct homological loops in Phase 0 Toy Validation.
        """
        if prob <= 0 or prob > 1.0:
            raise ValueError(f"Probability must be in (0, 1.0], got {prob}")
        
        key = (edge_type, t, x, y)
        if key not in self._edge_lookup:
            raise KeyError(f"Edge descriptor {key} not found in lattice geometry.")
        
        c_edge = self._edge_lookup[key]
        self.edge_weights[c_edge] = -math.log(prob)

    def get_weight(self, u: int, v: int) -> float:
        """Retrieves the assigned filtration weight for an edge pair."""
        return self.edge_weights.get(self._canonical_edge(u, v), self.default_bg_weight)

    def get_face_filtration_weight(self, face_vertices: Tuple[int, int, int, int]) -> float:
        """
        Determines the exact synchronization moment when a 2-cell face is completely bounded.
        By simplicial persistence monotonicity, a face is filled at the maximum weight of its boundary edges.
        """
        v0, v1, v2, v3 = face_vertices
        edges = [
            self._canonical_edge(v0, v1),
            self._canonical_edge(v1, v2),
            self._canonical_edge(v2, v3),
            self._canonical_edge(v3, v0)
        ]
        return max(self.edge_weights[e] for e in edges)

    def export_filtration_data(self, cone_weight: float = -1.0) -> Dict[str, List[Any]]:
        """
        Exports the entire filtered cellular pipeline fully prepared for SimplexTree assembly.
        Includes super-vertex cone edges to natively encode relative boundaries.
        """
        simplices = []
        
        # 1. Cone connections to relative boundary vertices (inserted earliest)
        for u, v in self.complex.relative_boundary_cone_edges():
            simplices.append({
                'vertices': [u, v],
                'weight': cone_weight,
                'dim': 1
            })

        # 2. Lattice edges
        for c_edge, weight in self.edge_weights.items():
            simplices.append({
                'vertices': list(c_edge),
                'weight': weight,
                'dim': 1
            })

        # 3. Lattice faces (triangulated dynamically to preserve quad geometry constraints)
        for face_data in self.complex.all_faces():
            v0, v1, v2, v3 = face_data['vertices']
            f_weight = self.get_face_filtration_weight((v0, v1, v2, v3))
            
            # To faithfully model a quad cell in a simplicial tree, we split it into two triangles.
            # We explicitly output the shared diagonal edge at the exact face closure weight.
            simplices.append({
                'vertices': sorted([v0, v2]),
                'weight': f_weight,
                'dim': 1
            })
            simplices.append({
                'vertices': sorted([v0, v1, v2]),
                'weight': f_weight,
                'dim': 2
            })
            simplices.append({
                'vertices': sorted([v0, v2, v3]),
                'weight': f_weight,
                'dim': 2
            })

        return {'simplices': simplices}
