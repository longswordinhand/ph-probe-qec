from typing import List, Tuple, Dict, Any

class SpacetimeComplex:
    """
    Represents the 3D Spacetime cell complex K for surface code persistent homology.
    
    Geometry:
    - Spatial dimensions: d x d with periodic (toric) boundary conditions.
    - Temporal dimension: T slices (t = 0 to T) with open boundary conditions.
    
    Vertices correspond to stabilizer measurement events in spacetime.
    Edges correspond to physical data qubit errors (spatial) or syndrome measurement errors (temporal).
    Faces (2-cells) enforce homological equivalence of local stabilizer configurations.
    """
    def __init__(self, d: int, T: int):
        self.d = d
        self.T = T
        self.num_grid_vertices = d * d * (T + 1)
        # Super vertex used to cone off the relative boundary \partial K
        self.super_vertex = self.num_grid_vertices

    def coord_to_id(self, x: int, y: int, t: int) -> int:
        """Maps spacetime grid coordinates to a unique integer vertex ID."""
        return x + y * self.d + t * self.d * self.d

    def id_to_coord(self, v_id: int) -> Tuple[int, int, int]:
        """Maps integer vertex ID back to spacetime coordinates (x, y, t)."""
        if v_id == self.super_vertex:
            raise ValueError("Super vertex does not have standard grid coordinates.")
        t = v_id // (self.d * self.d)
        rem = v_id % (self.d * self.d)
        y = rem // self.d
        x = rem % self.d
        return (x, y, t)

    def all_edges(self) -> List[Dict[str, Any]]:
        """
        Generates all 1-cells in the complex K.
        Returns a list of edge descriptors containing endpoints and topological classification.
        """
        edges = []
        # Spatial edges for each time slice
        for t in range(self.T + 1):
            for y in range(self.d):
                for x in range(self.d):
                    v_curr = self.coord_to_id(x, y, t)
                    
                    # Horizontal spatial edge (x-direction winding)
                    v_right = self.coord_to_id((x + 1) % self.d, y, t)
                    # To avoid graph multi-edges on d=1/2, ensure deterministic sorted pairs if needed,
                    # but for directed/labeled construction we maintain structural logic.
                    edges.append({
                        'type': 'spatial_h',
                        't': t, 'x': x, 'y': y,
                        'vertices': (v_curr, v_right)
                    })
                    
                    # Vertical spatial edge (y-direction winding)
                    v_up = self.coord_to_id(x, (y + 1) % self.d, t)
                    edges.append({
                        'type': 'spatial_v',
                        't': t, 'x': x, 'y': y,
                        'vertices': (v_curr, v_up)
                    })

        # Temporal edges connecting identical spatial locations across consecutive time slices
        for t in range(self.T):
            for y in range(self.d):
                for x in range(self.d):
                    v_curr = self.coord_to_id(x, y, t)
                    v_next = self.coord_to_id(x, y, t + 1)
                    edges.append({
                        'type': 'temporal',
                        't': t, 'x': x, 'y': y,
                        'vertices': (v_curr, v_next)
                    })
        return edges

    def all_faces(self) -> List[Dict[str, Any]]:
        """
        Generates all 2-cells (plaquettes) enforcing contractible stabilizer loops.
        Each face is returned as a sequence of 4 ordered vertices defining the boundary.
        """
        faces = []
        # 1. Spatial faces within each time slice
        for t in range(self.T + 1):
            for y in range(self.d):
                for x in range(self.d):
                    v0 = self.coord_to_id(x, y, t)
                    v1 = self.coord_to_id((x + 1) % self.d, y, t)
                    v2 = self.coord_to_id((x + 1) % self.d, (y + 1) % self.d, t)
                    v3 = self.coord_to_id(x, (y + 1) % self.d, t)
                    faces.append({
                        'type': 'spatial_face',
                        't': t,
                        'vertices': (v0, v1, v2, v3),
                        # Boundary edges mapped to ensure exact filtration synchronization
                        'boundary_edge_types': [
                            ('spatial_h', t, x, y),
                            ('spatial_v', t, (x + 1) % self.d, y),
                            ('spatial_h', t, x, (y + 1) % self.d),
                            ('spatial_v', t, x, y)
                        ]
                    })

        # 2. Spatiotemporal faces representing dynamic syndrome measurement repetition
        for t in range(self.T):
            for y in range(self.d):
                for x in range(self.d):
                    # Horizontal spatiotemporal face
                    vh0 = self.coord_to_id(x, y, t)
                    vh1 = self.coord_to_id((x + 1) % self.d, y, t)
                    vh2 = self.coord_to_id((x + 1) % self.d, y, t + 1)
                    vh3 = self.coord_to_id(x, y, t + 1)
                    faces.append({
                        'type': 'spatiotemporal_h',
                        't': t,
                        'vertices': (vh0, vh1, vh2, vh3)
                    })

                    # Vertical spatiotemporal face
                    vv0 = self.coord_to_id(x, y, t)
                    vv1 = self.coord_to_id(x, (y + 1) % self.d, t)
                    vv2 = self.coord_to_id(x, (y + 1) % self.d, t + 1)
                    vv3 = self.coord_to_id(x, y, t + 1)
                    faces.append({
                        'type': 'spatiotemporal_v',
                        't': t,
                        'vertices': (vv0, vv1, vv2, vv3)
                    })
        return faces

    def relative_boundary_vertices(self) -> List[int]:
        """
        Retrieves all vertices belonging to the temporal boundaries \partial K.
        Specifically, all spatial grid vertices at initial slice t=0 and final slice t=T.
        """
        bound_vertices = []
        for y in range(self.d):
            for x in range(self.d):
                bound_vertices.append(self.coord_to_id(x, y, 0))
                bound_vertices.append(self.coord_to_id(x, y, self.T))
        return bound_vertices

    def relative_boundary_cone_edges(self) -> List[Tuple[int, int]]:
        """
        Generates virtual edges connecting the super vertex to all relative boundary vertices.
        This provides the algebraic mechanism to compute relative homology H_1(K, \partial K)
        natively within absolute TDA solvers.
        """
        return [(v, self.super_vertex) for v in self.relative_boundary_vertices()]
