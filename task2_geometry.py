import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pyscf import gto, scf
from pyscf.geomopt import berny_solver

class Task2Geometry:
    def __init__(self, constants):
        self.constants = constants
    
    def create_molecule(self, atom_string, basis='sto-3g', spin=0):
        """Create a molecule object with proper spin state"""
        mol = gto.Mole()
        mol.atom = atom_string
        mol.basis = basis
        mol.unit = 'Angstrom'
        mol.spin = spin
        mol.build()
        return mol
    
    def run(self, shared_results):
        """T2: Optimize geometries with proper spin states"""
        print("Task 2: Geometry optimization")
        
        if 'task1' not in shared_results:
            print("Error: Task 1 must be completed first")
            return None
        
        geometries = shared_results['task1']
        optimized_geometries = {}
        optimization_data = []
        
        for mol_name, coord in geometries.items():
            print(f"Optimizing {mol_name}...")
            
            # Critical fix: Use triplet state for O2
            spin = 2 if mol_name == 'O2' else 0
            mol = self.create_molecule(coord, 'sto-3g', spin)
            
            # Use appropriate SCF method
            if mol_name == 'O2':
                mf = scf.UHF(mol)  # Unrestricted for open-shell O2
            else:
                mf = scf.RHF(mol)  # Restricted for closed-shell molecules
            
            try:
                mol_eq = berny_solver.optimize(mf)
                optimized_coord = mol_eq.atom
                optimized_geometries[mol_name] = optimized_coord
                
                # Calculate bond lengths with proper unit conversion
                coords = mol_eq.atom_coords()  # Returns coordinates in Bohr
                coords_ang = coords * 0.529177  # Convert Bohr to Angstrom
                
                # More precise experimental values
                if mol_name == 'H2':
                    bond_length = np.linalg.norm(coords_ang[0] - coords_ang[1])
                    exp_val = 0.74168
                    optimization_data.append({
                        'molecule': mol_name,
                        'property': 'H-H bond length',
                        'value': bond_length,
                        'experimental': exp_val,
                        'error': abs(bond_length - exp_val),
                        'percent_error': abs(bond_length - exp_val) / exp_val * 100
                    })
                elif mol_name == 'O2':
                    bond_length = np.linalg.norm(coords_ang[0] - coords_ang[1])
                    exp_val = 1.20752
                    optimization_data.append({
                        'molecule': mol_name,
                        'property': 'O-O bond length',
                        'value': bond_length,
                        'experimental': exp_val,
                        'error': abs(bond_length - exp_val),
                        'percent_error': abs(bond_length - exp_val) / exp_val * 100
                    })
                elif mol_name == 'H2O':
                    oh1 = np.linalg.norm(coords_ang[0] - coords_ang[1])
                    oh2 = np.linalg.norm(coords_ang[0] - coords_ang[2])
                    v1 = coords_ang[1] - coords_ang[0]
                    v2 = coords_ang[2] - coords_ang[0]
                    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                    angle_deg = np.degrees(angle)
                    
                    exp_bond = 0.95720  # NIST value
                    exp_angle = 104.474  # NIST value
                    
                    optimization_data.extend([
                        {'molecule': mol_name, 'property': 'O-H bond length 1', 'value': oh1, 'experimental': exp_bond, 'error': abs(oh1 - exp_bond), 'percent_error': abs(oh1 - exp_bond) / exp_bond * 100},
                        {'molecule': mol_name, 'property': 'O-H bond length 2', 'value': oh2, 'experimental': exp_bond, 'error': abs(oh2 - exp_bond), 'percent_error': abs(oh2 - exp_bond) / exp_bond * 100},
                        {'molecule': mol_name, 'property': 'H-O-H angle (deg)', 'value': angle_deg, 'experimental': exp_angle, 'error': abs(angle_deg - exp_angle), 'percent_error': abs(angle_deg - exp_angle) / exp_angle * 100}
                    ])
                    
            except Exception as e:
                print(f"Optimization failed for {mol_name}: {e}")
                continue
        
        # Enhanced visualization
        if optimization_data:
            opt_df = pd.DataFrame(optimization_data)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
            
            # Error comparison
            colors = ['red' if 'H2' in prop['molecule'] else 'blue' if 'O2' in prop['molecule'] else 'green' 
                     for prop in optimization_data]
            bars = ax1.bar(range(len(optimization_data)), opt_df['percent_error'], color=colors, alpha=0.7)
            ax1.set_xlabel('Molecular Property')
            ax1.set_ylabel('Percent Error (%)')
            ax1.set_title('HF/STO-3G vs Experimental Values', fontweight='bold')
            ax1.set_xticks(range(len(optimization_data)))
            ax1.set_xticklabels([f"{row['molecule']}\n{row['property'].split()[-1]}" for _, row in opt_df.iterrows()], 
                               rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Calculated vs experimental
            ax2.scatter(opt_df['experimental'], opt_df['value'], s=100, alpha=0.7, c=colors)
            min_val = min(opt_df['experimental'].min(), opt_df['value'].min())
            max_val = max(opt_df['experimental'].max(), opt_df['value'].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Agreement')
            ax2.set_xlabel('Experimental Value')
            ax2.set_ylabel('Calculated Value')
            ax2.set_title('Calculated vs Experimental', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Error distribution
            errors = opt_df['error']
            ax3.hist(errors, bins=5, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_xlabel('Absolute Error')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Error Distribution', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Percent error by molecule
            molecules = opt_df['molecule'].unique()
            mol_errors = [opt_df[opt_df['molecule'] == mol]['percent_error'].mean() for mol in molecules]
            ax4.bar(molecules, mol_errors, color=['red', 'blue', 'green'], alpha=0.7)
            ax4.set_xlabel('Molecule')
            ax4.set_ylabel('Average Percent Error (%)')
            ax4.set_title('Average Error by Molecule', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('plots/task2_geometry_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Save data
        if optimization_data:
            opt_df = pd.DataFrame(optimization_data)
            opt_df.to_csv('csv/task2_optimization.csv', index=False)
        
        with open('json/task2_optimization.json', 'w') as f:
            json.dump({
                'optimized_geometries': optimized_geometries,
                'analysis': optimization_data
            }, f, indent=2)
        
        print("Task 2 completed successfully!")
        return {
            'optimized_geometries': optimized_geometries,
            'analysis': optimization_data
        }
