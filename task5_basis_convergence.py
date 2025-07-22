import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pyscf import gto, scf

class Task5BasisConvergence:
    def __init__(self, constants):
        self.constants = constants
    
    def run(self, shared_results):
        """T5: Enhanced basis set convergence study"""
        print("Task 5: Basis set convergence study")
        
        if 'task2' not in shared_results:
            print("Error: Task 2 must be completed first")
            return None
        
        optimized_geometries = shared_results['task2']['optimized_geometries']
        basis_sets = ['sto-3g', '3-21g', '6-31g', 'cc-pvdz', 'cc-pvtz', 'cc-pvqz']
        convergence_data = []
        
        for basis in basis_sets:
            print(f"Calculating with basis set: {basis}")
            energies = {}
            
            try:
                for mol_name, coord in optimized_geometries.items():
                    spin = 2 if mol_name == 'O2' else 0
                    mol = gto.Mole()
                    mol.atom = coord
                    mol.basis = basis
                    mol.unit = 'Angstrom'
                    mol.spin = spin
                    mol.build()
                    
                    if mol_name == 'O2':
                        mf = scf.UHF(mol).run()
                    else:
                        mf = scf.RHF(mol).run()
                    
                    energies[mol_name] = mf.e_tot
                
                reaction_energy = (2 * energies['H2'] + energies['O2']) - (2 * energies['H2O'])
                reaction_energy_ev = reaction_energy * self.constants['HARTREE2EV']
                
                convergence_data.append({
                    'basis_set': basis,
                    'h2_energy': energies['H2'],
                    'o2_energy': energies['O2'],
                    'h2o_energy': energies['H2O'],
                    'reaction_energy_hartree': reaction_energy,
                    'reaction_energy_ev': reaction_energy_ev
                })
                
            except Exception as e:
                print(f"Error with basis {basis}: {e}")
                continue
        
        # Enhanced visualization
        self._create_convergence_plots(convergence_data)
        
        # Save data
        conv_df = pd.DataFrame(convergence_data)
        conv_df.to_csv('csv/task5_basis_convergence.csv', index=False)

        with open('json/task5_basis_convergence.json', 'w') as f:
            json.dump(convergence_data, f, indent=2)
        
        print("Task 5 completed successfully!")
        return convergence_data
    
    def _create_convergence_plots(self, convergence_data):
        """Create enhanced convergence visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
        
        basis_names = [d['basis_set'] for d in convergence_data]
        reaction_energies = [d['reaction_energy_ev'] for d in convergence_data]
        
        # Main convergence plot
        ax1.plot(basis_names, reaction_energies, 'bo-', linewidth=3, markersize=10)
        ax1.axhline(y=4.92, color='red', linestyle='--', linewidth=2, label='Experimental (4.92 eV)')
        ax1.set_xlabel('Basis Set')
        ax1.set_ylabel('Reaction Energy (eV)')
        ax1.set_title('Basis Set Convergence: Water Splitting Reaction', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        ax1.fill_between(basis_names, reaction_energies, alpha=0.3, color='blue')
        
        # Convergence rate
        if len(reaction_energies) > 1:
            differences = [abs(reaction_energies[i+1] - reaction_energies[i]) 
                          for i in range(len(reaction_energies)-1)]
            ax2.semilogy(basis_names[1:], differences, 'ro-', linewidth=2, markersize=8)
            ax2.set_xlabel('Basis Set')
            ax2.set_ylabel('|ΔE(n+1) - ΔE(n)| (eV)')
            ax2.set_title('Convergence Rate', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        
        # Individual molecule energies
        h2_energies = [d['h2_energy'] for d in convergence_data]
        o2_energies = [d['o2_energy'] for d in convergence_data]
        h2o_energies = [d['h2o_energy'] for d in convergence_data]
        
        ax3.plot(basis_names, h2_energies, 'g^-', label='H₂', linewidth=2, markersize=8)
        ax3.plot(basis_names, o2_energies, 'bs-', label='O₂', linewidth=2, markersize=8)
        ax3.plot(basis_names, h2o_energies, 'ro-', label='H₂O', linewidth=2, markersize=8)
        ax3.set_xlabel('Basis Set')
        ax3.set_ylabel('Energy (Hartree)')
        ax3.set_title('Individual Molecule Energies', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Error from experimental
        experimental_error = [abs(e - 4.92) for e in reaction_energies]
        bars = ax4.bar(basis_names, experimental_error, color='coral', alpha=0.7)
        ax4.set_xlabel('Basis Set')
        ax4.set_ylabel('Absolute Error (eV)')
        ax4.set_title('Error from Experimental Value', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # Add error values on bars
        for bar, error in zip(bars, experimental_error):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{error:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/task5_basis_convergence_enhanced.png', dpi=300, bbox_inches='tight')
        plt.show()
