"""
Task T5: Basis set convergence study
"""

import numpy as np
import pandas as pd
from config.constants import BASIS_SETS, HARTREE2EV
from task4_scf_energy import calculate_scf_energies
from utils.data_manager import data_manager

def calculate_reaction_energy_basis_series():
    """Calculate reaction energies across basis set series"""
    
    print("=== Task T5: Basis Set Convergence ===")
    
    results = {}
    csv_data = []
    
    for basis in BASIS_SETS:
        print(f"\nCalculating with {basis} basis...")
        
        try:
            energies = calculate_scf_energies(basis)
            
            # Calculate reaction energy
            delta_e = (2 * energies['H2'] + energies['O2']) - (2 * energies['H2O'])
            delta_e_ev = delta_e * HARTREE2EV
            
            results[basis] = {
                'energies': energies,
                'reaction_energy_hartree': delta_e,
                'reaction_energy_ev': delta_e_ev
            }
            
            print(f"  Reaction energy: {delta_e_ev:8.3f} eV")
            
            # Add to CSV data
            csv_data.append({
                'Basis_Set': basis,
                'H2_Energy_Hartree': energies['H2'],
                'O2_Energy_Hartree': energies['O2'],
                'H2O_Energy_Hartree': energies['H2O'],
                'Reaction_Energy_Hartree': delta_e,
                'Reaction_Energy_eV': delta_e_ev
            })
            
        except Exception as e:
            print(f"  Failed with {basis}: {e}")
            results[basis] = None
    
    # Print summary table
    print(f"\n{'Basis Set':<12} {'ΔE (eV)':<10}")
    print("-" * 25)
    for basis in BASIS_SETS:
        if results[basis] is not None:
            energy = results[basis]['reaction_energy_ev']
            print(f"{basis:<12} {energy:8.3f}")
        else:
            print(f"{basis:<12} {'Failed':<10}")
    
    # Create CSV table
    csv_table = pd.DataFrame(csv_data)
    
    # Save data
    data_manager.save_task_data(
        'task5_basis_convergence',
        results,
        {'basis_convergence': csv_table}
    )
    
    return results

if __name__ == "__main__":
    results = calculate_reaction_energy_basis_series()
