"""
Task T6: Compare single-molecule vs reaction energy convergence
"""

import numpy as np
import pandas as pd
from constants import BASIS_SETS, HARTREE2EV
from task5_basis_convergence import calculate_reaction_energy_basis_series
from data_manager import data_manager

def analyze_convergence_behavior():
    """Analyze convergence of H2O energy vs reaction energy"""
    
    print("=== Task T6: Convergence Comparison Analysis ===")
    
    # Get basis set data
    basis_results = calculate_reaction_energy_basis_series()
    
    # Extract H2O energies and reaction energies
    h2o_energies = []
    reaction_energies = []
    valid_basis = []
    csv_data = []
    
    for basis in BASIS_SETS:
        if basis_results[basis] is not None:
            h2o_energies.append(basis_results[basis]['energies']['H2O'])
            reaction_energies.append(basis_results[basis]['reaction_energy_ev'])
            valid_basis.append(basis)
    
    # Calculate convergence relative to largest basis
    if len(h2o_energies) > 1:
        h2o_ref = h2o_energies[-1]  # Use largest basis as reference
        reaction_ref = reaction_energies[-1]
        
        h2o_convergence = [(e - h2o_ref) * HARTREE2EV for e in h2o_energies]
        reaction_convergence = [e - reaction_ref for e in reaction_energies]
        
        print(f"\nConvergence relative to {valid_basis[-1]}:")
        print(f"{'Basis':<12} {'H2O (eV)':<12} {'Reaction (eV)':<12}")
        print("-" * 40)
        
        for i, basis in enumerate(valid_basis):
            print(f"{basis:<12} {h2o_convergence[i]:8.3f}    {reaction_convergence[i]:8.3f}")
            
            # Add to CSV data
            csv_data.append({
                'Basis_Set': basis,
                'H2O_Energy_Hartree': h2o_energies[i],
                'Reaction_Energy_eV': reaction_energies[i],
                'H2O_Convergence_eV': h2o_convergence[i],
                'Reaction_Convergence_eV': reaction_convergence[i],
                'Reference_Basis': valid_basis[-1]
            })
        
        # Analysis
        print(f"\nAnalysis:")
        if len(valid_basis) >= 3:
            h2o_change = abs(h2o_convergence[-2])
            reaction_change = abs(reaction_convergence[-2])
            
            if reaction_change < h2o_change:
                print("✓ Reaction energy converges faster than H2O total energy")
                print("  Reason: Basis set errors partially cancel between reactants and products")
            else:
                print("✗ H2O energy converges faster than reaction energy")
        
        results = {
            'basis_sets': valid_basis,
            'h2o_energies': h2o_energies,
            'reaction_energies': reaction_energies,
            'h2o_convergence': h2o_convergence,
            'reaction_convergence': reaction_convergence
        }
        
        # Create CSV table
        csv_table = pd.DataFrame(csv_data)
        
        # Save data
        data_manager.save_task_data(
            'task6_convergence_comparison',
            results,
            {'convergence_comparison': csv_table}
        )
        
        return results
    
    return None

if __name__ == "__main__":
    results = analyze_convergence_behavior()
