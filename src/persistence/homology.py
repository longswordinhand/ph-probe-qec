import gudhi
import numpy as np
from typing import List, Dict, Tuple, Any

class HomologyEngine:
    r"""
    Computes absolute and relative persistent homology using the GUDHI backend.
    
    By exploiting the Super Vertex Cone construction prepared in the filtration engine,
    relative cycles H_1(K, \partial K) are natively closed into absolute 1-cycles.
    Global logical operators are cleanly isolated by thresholding persistence lifetime,
    separating macroscopic global paths from microscopic local stabilizer fluctuations.
    """
    def __init__(self, st_complex, filtration_data: Dict[str, List[Any]], cone_weight: float = -1.0):
        self.complex = st_complex
        self.filtration_data = filtration_data
        self.cone_weight = cone_weight
        self.st = gudhi.SimplexTree()
        
        self._build_tree()

    def _build_tree(self):
        """
        Assembles the simplicial filtration tree.
        Ensures strict initial boundary condition mapping and monotonic face multi-cell triangulation.
        """
        # 1. Instantiate all 0-cells at the foundational boundary weight.
        # This guarantees that boundaries and spatial locations exist before any edge traversals.
        for v in range(self.complex.num_grid_vertices + 1):
            self.st.insert([v], filtration=self.cone_weight)

        # 2. Sequentially insert filtered edges and faces.
        # GUDHI automatically preserves internal structure and updates vertex bounds if necessary.
        for simplex in self.filtration_data['simplices']:
            self.st.insert(simplex['vertices'], filtration=simplex['weight'])

    def compute_persistence_intervals(self) -> List[Tuple[float, float]]:
        """
        Executes the persistent homology reduction algorithm.
        Returns all persistent intervals in dimension 1 (1-cycles).
        """
        # compute persistence with default non-destructive configuration
        self.st.persistence()
        intervals = self.st.persistence_intervals_in_dimension(1)
        # return as list of standard python floats for clean serialization
        return [(float(b), float(d)) for b, d in intervals]

    def extract_logical_bars(self, persistence_threshold: float = 1.0, cap_infinity: float = 10.0) -> List[Tuple[float, float]]:
        r"""
        Filters raw 1-cycle barcodes to isolate genuine logical topological sectors.
        
        Local stabilizer ring loops live short lives bounded by local plaquette closures (\ell \approx 0).
        Global logical cycles survive extensive parameter scales until background noise percolation.
        """
        raw_intervals = self.compute_persistence_intervals()
        logical_bars = []
        
        for birth, death in raw_intervals:
            # Handle topological generators that survive until the end of filtration
            d = death if death != float('inf') else cap_infinity
            persistence_length = d - birth
            
            if persistence_length > persistence_threshold:
                logical_bars.append((birth, d))
                
        return logical_bars

    def compute_observables(self, persistence_threshold: float = 1.0, cap_infinity: float = 10.0) -> Dict[str, Any]:
        r"""
        Computes the primary physical deliverables defined in the final research blueprint:
        - L_{logical}: Maximum logical persistence lifetime.
        - logical_bars: Raw birth-death distributions of topological sectors.
        - barcode_distribution: Length distributions \rho_{logical}(\ell).
        """
        bars = self.extract_logical_bars(persistence_threshold, cap_infinity)
        lengths = [d - b for b, d in bars]
        l_logical = max(lengths) if lengths else 0.0
        
        return {
            'L_logical': l_logical,
            'logical_bars': bars,
            'barcode_distribution': lengths
        }
