"""
Task T7: Jacob's ladder - method hierarchy
HF -> MP2 -> CCSD and PBE -> PBE0
"""

import pandas as pd
from pyscf import gto, scf, mp, cc, dft
from config.constants import HARTREE2EV, EXPERIMENTAL_WATER_SPLITTING_DG
from task1_coordinates import get_experimental_coordinates
from task2_geometry_optimization import build_molecule
from utils.data_manager import data_manager

def calculate_with_method(mol, method):
    """Calculate energy with specified method"""
    
    if method == 'HF':
        if mol.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
        mf.kernel()
        return mf.e_tot
    
    elif method == 'MP2':
        if mol.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
        mf.kernel()
        mp2 = mp.MP2(mf)
        mp2.kernel()
        return mp2.e_tot
    
    elif method == 'CCSD':
        if mol.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
        mf.kernel()
        ccsd = cc.CCSD(mf)
        ccsd.kernel()
        return ccsd.e_tot
    
    elif method == 'PBE':
        if mol.spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = 'PBE'
        mf.kernel()
        return mf.e_tot
    
    elif method == 'PBE0':
        if mol.spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = 'PBE0'
        mf.kernel()  
        return mf.e_tot
    
    else:
        raise ValueError(f"Method {method} not implemented")

def run_method_ladder():
    """Run Task T7: Method comparison"""
    
    print("=== Task T7: Jacob's Ladder - Method Hierarchy ===")
    
    coords = get_experimental_coordinates()
    basis = 'cc-pVDZ'  # Use converged basis from Task T5
    
    # Methods to test
    wavefunction_methods = ['HF', 'MP2', 'CCSD']
    dft_methods = ['PBE', 'PBE0']
    all_methods = wavefunction_methods + dft_methods
    
    results = {}
    csv_data = []
    
    for method in all_methods:
        print(f"\nCalculating with {method}...")
        
        try:
            energies = {}
            for molecule in ['H2', 'O2', 'H2O']:
                mol = build_molecule(molecule, coords[molecule], basis)
                energies[molecule] = calculate_with_method(mol, method)
                print(f"  {molecule}: {energies[molecule]:12.6f} Hartree")
            
            # Calculate reaction energy
            delta_e = (2 * energies['H2'] + energies['O2']) - (2 * energies['H2O'])
            delta_e_ev = delta_e * HARTREE2EV
            
            results[method] = {
                'energies': energies,
                'reaction_energy_ev': delta_e_ev,
                'error_vs_exp': delta_e_ev - EXPERIMENTAL_WATER_SPLITTING_DG
            }
            
            print(f"  Reaction energy: {delta_e_ev:8.3f} eV")
            print(f"  Error vs exp: {delta_e_ev - EXPERIMENTAL_WATER_SPLITTING_DG:+8.3f} eV")
            
            # Add to CSV data
            csv_data.append({
                'Method': method,
                'H2_Energy_Hartree': energies['H2'],
                'O2_Energy_Hartree': energies['O2'],
                'H2O_Energy_Hartree': energies['H2O'],
                'Reaction_Energy_eV': delta_e_ev,
                'Error_vs_Exp_eV': delta_e_ev - EXPERIMENTAL_WATER_SPLITTING_DG,
                'Experimental_Reference_eV': EXPERIMENTAL_WATER_SPLITTING_DG
            })
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[method] = None
    
    # Summary table
    print(f"\n{'Method':<8} {'ΔE (eV)':<10} {'Error (eV)':<10}")
    print("-" * 35)
    for method in all_methods:
        if results[method] is not None:
            energy = results[method]['reaction_energy_ev']
            error = results[method]['error_vs_exp']
            print(f"{method:<8} {energy:8.3f}  {error:+8.3f}")
    
    print(f"\nExperimental reference: {EXPERIMENTAL_WATER_SPLITTING_DG:.2f} eV")
    
    # Create CSV table
    csv_table = pd.DataFrame(csv_data)
    
    # Save data
    data_manager.save_task_data(
        'task7_method_ladder',
        results,
        {'method_ladder': csv_table}
    )
    
    return results

if __name__ == "__main__":
    results = run_method_ladder()
