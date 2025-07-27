"""
Task T8: Add thermodynamic corrections (ZPE, thermal enthalpy, entropy)
CORRECTED VERSION - Fixed entropy data and unit conversions
"""

import pandas as pd
from config.constants import (NIST_THERMO_DATA, KCAL2HARTREE, R_GAS, STANDARD_TEMP, 
                      HARTREE2EV, EXPERIMENTAL_WATER_SPLITTING_DG, HARTREE2KCAL)
from task7_method_ladder import run_method_ladder
from utils.data_manager import data_manager

def calculate_thermodynamic_corrections():
    """Add NIST thermodynamic corrections to electronic energies - FIXED VERSION"""
    
    print("=== Task T8: Thermodynamic Corrections ===")
    # Get electronic energies from best method (MP2)
    method_results = run_method_ladder()
    mp2_results = method_results['MP2']
    
    if mp2_results is None:
        print("MP2 calculation failed, cannot proceed with thermodynamics")
        return None

    print(f"\nApplying CORRECTED NIST thermodynamic corrections at {STANDARD_TEMP} K:")
    print("FIXES APPLIED:")
    print("- Correct NIST entropy values (H₂: 130.68, O₂: 205.152, H₂O_liq: 69.95 J mol⁻¹ K⁻¹)")
    print("- Proper unit conversion: J mol⁻¹ K⁻¹ → Hartree K⁻¹") 
    print("- Eliminated compound conversion errors")
    
    # CORRECTED NIST thermodynamic data with proper values
    CORRECTED_NIST_DATA = {
        'H2': {
            'entropy_gas': 130.68,        # J mol⁻¹ K⁻¹ (CODATA)
            'zpe': 6.197,                 # kcal mol⁻¹ 
            'thermal_enthalpy': 2.024     # kcal mol⁻¹
        },
        'O2': {
            'entropy_gas': 205.152,       # J mol⁻¹ K⁻¹ (CODATA) 
            'zpe': 0.988,                 # kcal mol⁻¹
            'thermal_enthalpy': 2.024     # kcal mol⁻¹
        },
        'H2O': {
            'entropy_liquid': 69.95,      # J mol⁻¹ K⁻¹ (liquid water at 298K)
            'entropy_gas': 188.84,        # J mol⁻¹ K⁻¹ (gas phase)
            'zpe': 13.435,                # kcal mol⁻¹
            'thermal_enthalpy': 2.368     # kcal mol⁻¹
        }
    }
    
    corrected_energies = {}
    csv_breakdown = []
    csv_components = []
    
    for molecule in ['H2', 'O2', 'H2O']:
        electronic_energy = mp2_results['energies'][molecule]
        nist_data = CORRECTED_NIST_DATA[molecule]
        
        # Zero-point energy correction (kcal/mol → Hartree)
        zpe_correction = nist_data['zpe'] * KCAL2HARTREE
        
        # Thermal enthalpy correction (kcal/mol → Hartree)
        thermal_correction = nist_data['thermal_enthalpy'] * KCAL2HARTREE
        
        # FIXED ENTROPY CORRECTION: -T*S
        if molecule == 'H2O':
            # Use liquid entropy for water at 298K (reaction involves liquid water)
            entropy_j_mol_k = nist_data['entropy_liquid']
        else:
            entropy_j_mol_k = nist_data['entropy_gas']
        
        # CORRECTED unit conversion: J mol⁻¹ K⁻¹ → Hartree K⁻¹
        # Method: J mol⁻¹ K⁻¹ → cal mol⁻¹ K⁻¹ → kcal mol⁻¹ K⁻¹ → Hartree K⁻¹
        entropy_cal_mol_k = entropy_j_mol_k / 4.184        # J → cal
        entropy_kcal_mol_k = entropy_cal_mol_k / 1000      # cal → kcal  
        entropy_hartree_per_k = entropy_kcal_mol_k * KCAL2HARTREE  # kcal → Hartree
        
        # Final entropy correction: -T*S
        entropy_correction = -STANDARD_TEMP * entropy_hartree_per_k
        
        # Total corrected free energy: G = E + ZPE + H_thermal - T*S
        g_corrected = electronic_energy + zpe_correction + thermal_correction + entropy_correction
        
        corrected_energies[molecule] = {
            'electronic': electronic_energy,
            'zpe': zpe_correction,
            'thermal': thermal_correction,
            'entropy': entropy_correction,
            'total_g': g_corrected
        }
        
        print(f"\n{molecule} (CORRECTED):")
        print(f"  Electronic:     {electronic_energy:12.6f} Hartree")
        print(f"  ZPE:            {zpe_correction:+12.6f} Hartree")
        print(f"  Thermal H:      {thermal_correction:+12.6f} Hartree") 
        print(f"  -T*S:           {entropy_correction:+12.6f} Hartree ({entropy_correction * HARTREE2EV:+8.3f} eV)")
        print(f"  Total G:        {g_corrected:12.6f} Hartree")
        print(f"  Entropy (NIST): {entropy_j_mol_k:8.2f} J mol⁻¹ K⁻¹")
        
        # Add to CSV breakdown
        csv_breakdown.append({
            'Molecule': molecule,
            'Electronic_Hartree': electronic_energy,
            'Electronic_eV': electronic_energy * HARTREE2EV,
            'ZPE_Hartree': zpe_correction,
            'ZPE_eV': zpe_correction * HARTREE2EV,
            'Thermal_H_Hartree': thermal_correction,
            'Thermal_H_eV': thermal_correction * HARTREE2EV,
            'Entropy_Hartree': entropy_correction,
            'Entropy_eV': entropy_correction * HARTREE2EV,
            'Total_G_Hartree': g_corrected,
            'Total_G_eV': g_corrected * HARTREE2EV,
            'NIST_Entropy_J_mol_K': entropy_j_mol_k
        })
    
    # Calculate CORRECTED reaction free energy: 2H₂O(l) → 2H₂(g) + O₂(g)
    delta_g_hartree = (2 * corrected_energies['H2']['total_g'] + 
                      corrected_energies['O2']['total_g'] - 
                      2 * corrected_energies['H2O']['total_g'])
    
    delta_g_ev = delta_g_hartree * HARTREE2EV
    error_vs_exp = delta_g_ev - EXPERIMENTAL_WATER_SPLITTING_DG
    
    print(f"\n{'='*60}")
    print(f"CORRECTED THERMODYNAMIC RESULTS:")
    print(f"{'='*60}")
    print(f"Reaction: 2H₂O(l) → 2H₂(g) + O₂(g)")
    print(f"Electronic ΔE:        {mp2_results['reaction_energy_ev']:8.3f} eV")
    print(f"Thermodynamic ΔG:     {delta_g_ev:8.3f} eV")
    print(f"Experimental ΔG:      {EXPERIMENTAL_WATER_SPLITTING_DG:8.3f} eV")
    print(f"Error:                {error_vs_exp:+8.3f} eV ({abs(error_vs_exp)/EXPERIMENTAL_WATER_SPLITTING_DG*100:.1f}%)")
    
    # Performance assessment
    if abs(error_vs_exp) < 0.043:  # 1 kcal/mol
        performance = "CHEMICAL ACCURACY"
    elif abs(error_vs_exp) < 0.1:
        performance = "EXCELLENT"  
    elif abs(error_vs_exp) < 0.3:
        performance = "GOOD"
    else:
        performance = "FAIR"
    
    print(f"Performance:          {performance}")
    print(f"{'='*60}")
    
    # Add reaction components to CSV
    csv_components.extend([
        {
            'Component': 'Electronic_ΔE',
            'Value_eV': mp2_results['reaction_energy_ev'],
            'Description': 'Pure electronic reaction energy'
        },
        {
            'Component': 'Thermodynamic_ΔG_CORRECTED',
            'Value_eV': delta_g_ev,
            'Description': 'Including ZPE, thermal, and CORRECTED entropy'
        },
        {
            'Component': 'Experimental_ΔG',
            'Value_eV': EXPERIMENTAL_WATER_SPLITTING_DG,
            'Description': 'Literature reference value'
        },
        {
            'Component': 'Error_CORRECTED',
            'Value_eV': error_vs_exp,
            'Description': 'Calculated - Experimental (FIXED)'
        }
    ])
    
    results = {
        'electronic_results': mp2_results,
        'corrected_energies': corrected_energies,
        'reaction_delta_g_ev': delta_g_ev,
        'error_vs_experimental': error_vs_exp,
        'performance_rating': performance,
        'corrections_applied': [
            'Fixed NIST entropy values',
            'Corrected unit conversions', 
            'Proper liquid water entropy',
            'Eliminated conversion errors'
        ]
    }
    
    # Create CSV tables
    breakdown_table = pd.DataFrame(csv_breakdown)
    components_table = pd.DataFrame(csv_components)
    
    # Save data
    data_manager.save_task_data(
        'task8_thermodynamics_CORRECTED',
        results,
        {
            'thermodynamic_breakdown_CORRECTED': breakdown_table,
            'reaction_components_CORRECTED': components_table
        }
    )
    
    return results

if __name__ == "__main__":
    results = calculate_thermodynamic_corrections()
