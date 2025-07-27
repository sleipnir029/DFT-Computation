"""
Task T3: Calculate reaction energy for water splitting: 2H2O -> 2H2 + O2
"""

import pandas as pd
from config.constants import HARTREE2EV
from task2_geometry_optimization import run_geometry_optimization
from utils.data_manager import data_manager

def calculate_reaction_energy(energies):
    """
    Calculate reaction energy: 2H2O -> 2H2 + O2
    
    Args:
        energies: Dictionary with molecular energies in Hartree
        
    Returns:
        float: Reaction energy in eV
    """
    
    # Reaction energy = Products - Reactants
    # ΔE = (2*E_H2 + E_O2) - 2*E_H2O
    delta_e_hartree = (2 * energies['H2'] + energies['O2']) - (2 * energies['H2O'])
    delta_e_ev = delta_e_hartree * HARTREE2EV
    
    return delta_e_ev

def run_reaction_energy_calculation():
    """Run Task T3: Calculate reaction energy"""
    
    print("=== Task T3: Reaction Energy Calculation ===")
    
    # Get optimized geometries and energies from Task T2
    optimization_results = run_geometry_optimization()
    
    # Extract energies
    energies = {}
    csv_data = []
    
    for molecule in ['H2', 'O2', 'H2O']:
        energies[molecule] = optimization_results[molecule]['energy']
        print(f"{molecule} energy: {energies[molecule]:12.6f} Hartree")
        
        # Add to CSV data
        csv_data.append({
            'Molecule': molecule,
            'Energy_Hartree': energies[molecule],
            'Energy_eV': energies[molecule] * HARTREE2EV
        })
    
    # Calculate reaction energy
    reaction_energy = calculate_reaction_energy(energies)
    
    # Add reaction energy to CSV
    csv_data.append({
        'Molecule': 'Reaction',
        'Energy_Hartree': reaction_energy / HARTREE2EV,
        'Energy_eV': reaction_energy
    })
    
    print(f"\nReaction: 2H2O -> 2H2 + O2")
    print(f"Electronic reaction energy: {reaction_energy:8.3f} eV")
    
    results = {
        'energies': energies,
        'reaction_energy_ev': reaction_energy,
        'optimization_results': optimization_results
    }
    
    # Create CSV table
    csv_table = pd.DataFrame(csv_data)
    
    # Save data
    data_manager.save_task_data(
        'task3_reaction_energy',
        results,
        {'reaction_energy': csv_table}
    )
    
    return results

if __name__ == "__main__":
    results = run_reaction_energy_calculation()
