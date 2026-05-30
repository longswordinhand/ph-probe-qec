import numpy as np
from src.geometry.spacetime import SpacetimeComplex
from src.syndrome.noise import NoiseModel
from src.decoding.baseline import MWPMBaselineDecoder

def test_mwpm_single_fault_correction():
    r"""
    Correction Success Verification:
    Asserts that localized, topologically contractible multi-qubit physical errors are cleanly
    resolved by MWPM baseline matching without flipping macroscopic homological parities.
    """
    complex_lattice = SpacetimeComplex(d=3, T=2)
    p_base = 0.01
    
    noise = NoiseModel(complex_lattice, p_spatial=p_base, p_temporal=p_base)
    mwpm = MWPMBaselineDecoder(complex_lattice, p_spatial=p_base, p_temporal=p_base)
    
    # Inject simple highly contractible localized spatial fault pair
    target_desc = [('spatial_h', 0, 0, 0)]
    faults = noise.generate_custom_fault(target_desc)
    s_obs = noise.compute_syndrome(faults)
    
    fail_x, fail_y = mwpm.decode_syndrome(s_obs, faults)
    # Conventional decoder successfully unifies open bounds without error loops
    assert not fail_x and not fail_y

def test_mwpm_topological_breakdown():
    r"""
    Topological Breakdown Verification:
    Asserts that string physical errors spanning fully over spatial dimensions trigger guaranteed
    macro homological failures when classical MWPM makes misaligned classical assumptions.
    """
    complex_lattice = SpacetimeComplex(d=3, T=2)
    p_base = 0.01
    
    noise = NoiseModel(complex_lattice, p_spatial=p_base, p_temporal=p_base)
    mwpm = MWPMBaselineDecoder(complex_lattice, p_spatial=p_base, p_temporal=p_base)
    
    # Form macro non-contractible spatial string spanning fully across horizontal axis at t=0, y=1
    target_desc = [('spatial_h', 0, x, 1) for x in range(complex_lattice.d)]
    faults = noise.generate_custom_fault(target_desc)
    s_obs = noise.compute_syndrome(faults)
    
    # Since total loop exactly completes zero boundary syndrome, classical graph observes vacuum.
    # MWPM leaves actual faults unmodified. Merging configuration forces absolute logical parity flip.
    fail_x, fail_y = mwpm.decode_syndrome(s_obs, faults)
    assert fail_x or fail_y

if __name__ == '__main__':
    print("Executing Phase 1 Comparative Decoding Baseline Assertions...\n")
    
    print("1. Testing Classic MWPM Localized Single-Fault Resolution...")
    test_mwpm_single_fault_correction()
    print(" -> Localized physical errors successfully unmerged without triggering macro failure.\n")
    
    print("2. Testing Conventional MWPM Macro String Breakdown Tracking...")
    test_mwpm_topological_breakdown()
    print(" -> Non-contractible macro fault chains rigorously flagged as absolute logical failures.\n")
    
    print("==================================================================")
    print("🎉 ALL PHASE 1 DECODING BASELINE ASSERTIONS PASSED WITH ABSOLUTE FIDELITY!")
    print("==================================================================")
