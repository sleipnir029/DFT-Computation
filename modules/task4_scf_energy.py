"""
Task T4: Record SCF total energies before basis set study
"""

import pandas as pd
from pyscf import gto, scf
from task1_coordinates import get_experimental_coordinates
from task2_geometry_optimization import build_molecule
from utils.data_manager import data_manager
from config.constants import HARTREE2EV

def calculate_scf_energies(basis='STO-3G'):
    """Calculate SCF energies for all molecules with given basis"""
    
    coords = get_experimental_coordinates()
    energies = {}
    
    for molecule in ['H2', 'O2', 'H2O']:
        # Use experimental geometries (not optimized)
        mol = build_molecule(molecule, coords[molecule], basis)
        
        # SCF calculation
        if mol.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
        
        mf.kernel()
        energies[molecule] = mf.e_tot
    
    return energies

def run_scf_energy_recording():
    """Run Task T4: Record SCF energies"""
    
    print("=== Task T4: SCF Total Energies ===")
    
    energies = calculate_scf_energies()
    csv_data = []
    
    print("Hartree-Fock energies with STO-3G basis:")
    for molecule, energy in energies.items():
        print(f"{molecule:>4}: {energy:12.6f} Hartree")
        
        # Add to CSV data
        csv_data.append({
            'Molecule': molecule,
            'Basis': 'STO-3G',
            'Method': 'HF',
            'Energy_Hartree': energy,
            'Energy_eV': energy * HARTREE2EV
        })
    
    # Create CSV table
    csv_table = pd.DataFrame(csv_data)
    
    # Save data
    data_manager.save_task_data(
        'task4_scf_energy',
        energies,
        {'scf_energies': csv_table}
    )
    
    return energies

if __name__ == "__main__":
    results = run_scf_energy_recording()
