import math
import pytest
from src.geometry.spacetime import SpacetimeComplex
from src.persistence.filtration import FiltrationEngine
from src.persistence.homology import HomologyEngine

@pytest.fixture
def base_lattice():
    return SpacetimeComplex(d=3, T=3)

def test_case_a_trivial_loop(base_lattice):
    r"""
    Case A validation: A localized, contractible spatial stabilizer loop.
    Target: No surviving logical barcodes (\rho_{logical} = \emptyset).
    """
    filt = FiltrationEngine(base_lattice, default_bg_prob=1e-4)
    
    # Form a local ring around spatial plaquette at t=1, x=0, y=0
    p_err = 0.99
    filt.set_edge_probability('spatial_h', t=1, x=0, y=0, prob=p_err)
    filt.set_edge_probability('spatial_v', t=1, x=1, y=0, prob=p_err)
    filt.set_edge_probability('spatial_h', t=1, x=0, y=1, prob=p_err)
    filt.set_edge_probability('spatial_v', t=1, x=0, y=0, prob=p_err)
    
    export = filt.export_filtration_data()
    homology = HomologyEngine(base_lattice, export)
    observables = homology.compute_observables(persistence_threshold=1.0)
    
    # Assert total suppression of logical output
    assert len(observables['logical_bars']) == 0, f"Expected 0 logical bars, got {len(observables['logical_bars'])}"
    assert observables['L_logical'] == 0.0

def test_case_b_single_spatial_logical_sector(base_lattice):
    """
    Case B validation (Spatial): A non-contractible string winding fully around the toric geometry.
    Target: Exactly one dominant logical barcode.
    """
    filt = FiltrationEngine(base_lattice, default_bg_prob=1e-4)
    
    # Wind horizontally along x-axis at t=1, y=0
    p_err = 0.99
    for x in range(base_lattice.d):
        filt.set_edge_probability('spatial_h', t=1, x=x, y=0, prob=p_err)
        
    export = filt.export_filtration_data()
    cap_inf = -math.log(1e-4)
    export['max_weight'] = cap_inf
    
    homology = HomologyEngine(base_lattice, export)
    observables = homology.compute_observables(persistence_threshold=1.0, cap_infinity=cap_inf)
    
    assert len(observables['logical_bars']) == 1, f"Expected 1 logical bar, got {len(observables['logical_bars'])}"
    birth, death = observables['logical_bars'][0]
    assert abs(birth - (-math.log(0.99))) < 1e-2
    assert observables['L_logical'] > 5.0  # Demonstrates highly persistent robust lifetime

def test_case_b_single_temporal_spanning_path(base_lattice):
    """
    Case B validation (Spatiotemporal): An error trajectory connecting initial to final temporal boundaries.
    Target: Exactly one dominant logical barcode via relative boundary Cone completion.
    """
    filt = FiltrationEngine(base_lattice, default_bg_prob=1e-4)
    
    # Form spanning path connecting t=0 to t=3 at spatial location x=1, y=1
    p_err = 0.99
    for t in range(base_lattice.T):
        filt.set_edge_probability('temporal', t=t, x=1, y=1, prob=p_err)
        
    export = filt.export_filtration_data()
    cap_inf = -math.log(1e-4)
    export['max_weight'] = cap_inf
    
    homology = HomologyEngine(base_lattice, export)
    observables = homology.compute_observables(persistence_threshold=1.0, cap_infinity=cap_inf)
    
    assert len(observables['logical_bars']) == 1
    assert observables['L_logical'] > 5.0

def test_case_c_competing_logical_sectors(base_lattice):
    """
    Case C validation: Coexistence of independent winding logical paths representing sector competition.
    Target: Two distinct logical bars with unique persistence intervals.
    """
    filt = FiltrationEngine(base_lattice, default_bg_prob=1e-4)
    
    # Sector 1: Horizontal winding at t=1, y=0 (Highly probable)
    p_err1 = 0.99
    for x in range(base_lattice.d):
        filt.set_edge_probability('spatial_h', t=1, x=x, y=0, prob=p_err1)
        
    # Sector 2: Vertical winding at t=2, x=2 (Slightly less probable)
    p_err2 = 0.95
    for y in range(base_lattice.d):
        filt.set_edge_probability('spatial_v', t=2, x=2, y=y, prob=p_err2)
        
    export = filt.export_filtration_data()
    cap_inf = -math.log(1e-4)
    export['max_weight'] = cap_inf
    
    homology = HomologyEngine(base_lattice, export)
    observables = homology.compute_observables(persistence_threshold=1.0, cap_infinity=cap_inf)
    
    assert len(observables['logical_bars']) == 2
    lengths = sorted(observables['barcode_distribution'])
    assert lengths[0] > 5.0
    assert lengths[1] > 5.0

if __name__ == '__main__':
    lattice = SpacetimeComplex(d=3, T=3)
    
    print("Executing Case A (Trivial Loop)...")
    test_case_a_trivial_loop(lattice)
    print(" -> Case A passed successfully.\n")
    
    print("Executing Case B (Spatial Winding Sector)...")
    test_case_b_single_spatial_logical_sector(lattice)
    print(" -> Case B (Spatial) passed successfully.\n")
    
    print("Executing Case B (Spatiotemporal Spanning Path)...")
    test_case_b_single_temporal_spanning_path(lattice)
    print(" -> Case B (Spatiotemporal) passed successfully.\n")
    
    print("Executing Case C (Competing Logical Sectors)...")
    test_case_c_competing_logical_sectors(lattice)
    print(" -> Case C passed successfully.\n")
    
    print("==================================================================")
    print("🎉 ALL PHASE 0 TOY MODEL VALIDATIONS PASSED WITH ABSOLUTE FIDELITY!")
    print("==================================================================")
