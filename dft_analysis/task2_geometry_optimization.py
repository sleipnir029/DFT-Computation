"""
Task T2: Geometry optimization using HF/STO-3G
"""

import numpy as np
import pandas as pd
from pyscf import gto, scf

try:
    from pyscf.geomopt import berny_solver
    optimize = berny_solver.optimize
    opt_name = "Berny"
except (ModuleNotFoundError, ImportError):
    try:
        from pyscf.geomopt import geometric_solver
        optimize = geometric_solver.optimize
        opt_name = "geomeTRIC"
    except (ModuleNotFoundError, ImportError):
        print("Warning: No geometry optimizer available")
        optimize = None
        opt_name = "None"

from constants import EXPERIMENTAL_GEOMETRIES, EXPERIMENTAL_BOND_LENGTHS, EXPERIMENTAL_BOND_ANGLES
from task1_coordinates import get_experimental_coordinates
from data_manager import data_manager

def build_molecule(molecule_name, coords_data, basis='STO-3G'):
    """Build PySCF molecule object"""
    
    atom_string = []
    for atom, coord in zip(coords_data['atoms'], coords_data['coordinates']):
        atom_string.append(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
    
    mol = gto.Mole()
    mol.atom = '; '.join(atom_string)
    mol.basis = basis
    mol.charge = coords_data['charge']
    mol.spin = coords_data['spin']
    mol.unit = 'Angstrom'
    mol.build()
    
    return mol

def optimize_geometry(mol, method='HF'):
    """Optimize geometry using specified method"""
    
    if method == 'HF':
        if mol.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
    else:
        raise ValueError(f"Method {method} not implemented yet")
    
    # Initial SCF
    mf.kernel()
    
    # Geometry optimization
    if optimize is not None:
        try:
            mol_eq = optimize(mf)
            return mol_eq, mf.e_tot
        except Exception as e:
            print(f"Optimization failed: {e}")
            return mol, mf.e_tot
    else:
        print("No optimizer available, using initial geometry")
        return mol, mf.e_tot

def calculate_bond_properties(mol):
    """Calculate bond lengths and angles from optimized geometry"""
    
    coords = mol.atom_coords() * 0.529177249  # Bohr to Angstrom
    atoms = [atom[0] for atom in mol._atom]
    
    properties = {}
    
    if len(atoms) == 2:  # Diatomic
        bond_length = np.linalg.norm(coords[1] - coords[0])
        properties['bond_length'] = bond_length
        
    elif len(atoms) == 3 and atoms == ['O', 'H', 'H']:  # Water
        # O-H bond lengths
        oh1 = np.linalg.norm(coords[1] - coords[0])
        oh2 = np.linalg.norm(coords[2] - coords[0])
        properties['oh1_length'] = oh1
        properties['oh2_length'] = oh2
        properties['oh_avg_length'] = (oh1 + oh2) / 2
        
        # H-O-H angle
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        properties['hoh_angle'] = angle
    
    return properties

def run_geometry_optimization():
    """Run Task T2: Geometry optimization with HF/STO-3G"""
    
    print("=== Task T2: Geometry Optimization (HF/STO-3G) ===")
    
    coords = get_experimental_coordinates()
    results = {}
    csv_data = []
    
    for molecule in ['H2', 'O2', 'H2O']:
        print(f"\nOptimizing {molecule}...")
        
        # Build molecule
        mol = build_molecule(molecule, coords[molecule])
        
        # Optimize
        mol_opt, energy = optimize_geometry(mol)
        
        # Calculate properties
        properties = calculate_bond_properties(mol_opt)
        
        # Compare with experiment
        comparisons = {}
        if molecule == 'H2':
            exp_length = EXPERIMENTAL_BOND_LENGTHS['H2']
            calc_length = properties['bond_length']
            error = calc_length - exp_length
            comparisons['bond_length'] = {
                'experimental': exp_length,
                'calculated': calc_length,
                'error': error,
                'percent_error': abs(error) / exp_length * 100
            }
            
            # Add to CSV data
            csv_data.append({
                'Molecule': molecule,
                'Property': 'Bond_Length',
                'Experimental': exp_length,
                'Calculated': calc_length,
                'Error': error,
                'Percent_Error': abs(error) / exp_length * 100,
                'Units': 'Angstrom'
            })
            
        elif molecule == 'O2':
            exp_length = EXPERIMENTAL_BOND_LENGTHS['O2']
            calc_length = properties['bond_length']
            error = calc_length - exp_length
            comparisons['bond_length'] = {
                'experimental': exp_length,
                'calculated': calc_length,
                'error': error,
                'percent_error': abs(error) / exp_length * 100
            }
            
            # Add to CSV data
            csv_data.append({
                'Molecule': molecule,
                'Property': 'Bond_Length',
                'Experimental': exp_length,
                'Calculated': calc_length,
                'Error': error,
                'Percent_Error': abs(error) / exp_length * 100,
                'Units': 'Angstrom'
            })
            
        elif molecule == 'H2O':
            # Bond length comparison
            exp_oh = EXPERIMENTAL_BOND_LENGTHS['H2O_OH']
            calc_oh = properties['oh_avg_length']
            bond_error = calc_oh - exp_oh
            
            # Bond angle comparison
            exp_angle = EXPERIMENTAL_BOND_ANGLES['H2O_HOH']
            calc_angle = properties['hoh_angle']
            angle_error = calc_angle - exp_angle
            
            comparisons['oh_bond_length'] = {
                'experimental': exp_oh,
                'calculated': calc_oh,
                'error': bond_error,
                'percent_error': abs(bond_error) / exp_oh * 100
            }
            comparisons['hoh_angle'] = {
                'experimental': exp_angle,
                'calculated': calc_angle,
                'error': angle_error,
                'percent_error': abs(angle_error) / exp_angle * 100
            }
            
            # Add to CSV data
            csv_data.extend([
                {
                    'Molecule': molecule,
                    'Property': 'OH_Bond_Length',
                    'Experimental': exp_oh,
                    'Calculated': calc_oh,
                    'Error': bond_error,
                    'Percent_Error': abs(bond_error) / exp_oh * 100,
                    'Units': 'Angstrom'
                },
                {
                    'Molecule': molecule,
                    'Property': 'HOH_Angle',
                    'Experimental': exp_angle,
                    'Calculated': calc_angle,
                    'Error': angle_error,
                    'Percent_Error': abs(angle_error) / exp_angle * 100,
                    'Units': 'Degrees'
                }
            ])
        
        results[molecule] = {
            'energy': energy,
            'properties': properties,
            'comparisons': comparisons
        }
        
        print(f"  Final energy: {energy:.6f} Hartree")
        if 'bond_length' in properties:
            print(f"  Bond length: {properties['bond_length']:.3f} Å")
        if 'oh_avg_length' in properties:
            print(f"  Average O-H length: {properties['oh_avg_length']:.3f} Å")
            print(f"  H-O-H angle: {properties['hoh_angle']:.1f}°")
    
    # Create CSV table
    csv_table = pd.DataFrame(csv_data)
    
    # Save data
    data_manager.save_task_data(
        'task2_geometry_optimization',
        results,
        {'geometry_optimization': csv_table}
    )
    
    return results

if __name__ == "__main__":
    results = run_geometry_optimization()
