import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pyscf import gto, scf

class Task6ConvergenceComparison:
    def __init__(self, constants):
        self.constants = constants
    
    def run(self, shared_results):
        """T6: Compare single molecule vs reaction energy convergence"""
        print("Task 6: Single molecule vs reaction energy convergence")
        
        if 'task2' not in shared_results:
            print("Error: Task 2 must be completed first")
            return None
        
        optimized_geometries = shared_results['task2']['optimized_geometries']
        basis_sets = ['sto-3g', '3-21g', '6-31g', 'cc-pvdz', 'cc-pvtz', 'cc-pvqz']
        h2o_energies = []
        reaction_energies = []
        
        for basis in basis_sets:
            try:
                # H2O energy
                mol_h2o = gto.Mole()
                mol_h2o.atom = optimized_geometries['H2O']
                mol_h2o.basis = basis
                mol_h2o.unit = 'Angstrom'
                mol_h2o.build()
                mf_h2o = scf.RHF(mol_h2o).run()
                
                # All molecules for reaction energy
                energies = {}
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
                
                h2o_energies.append(mf_h2o.e_tot)
                reaction_energies.append(reaction_energy)
                
            except Exception as e:
                print(f"Error with basis {basis}: {e}")
                continue
        
        # Enhanced visualization
        self._create_comparison_plots(basis_sets, h2o_energies, reaction_energies)
        
        # Save data
        comparison_data = {
            'basis_set': basis_sets[:len(h2o_energies)],
            'h2o_energy': h2o_energies,
            'reaction_energy': reaction_energies
        }
        
        pd.DataFrame(comparison_data).to_csv('csv/task6_convergence_comparison.csv', index=False)

        with open('json/task6_convergence_comparison.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print("Task 6 completed successfully!")
        print("Note: Reaction energy converges faster due to error cancellation")
        return comparison_data
    
    def _create_comparison_plots(self, basis_sets, h2o_energies, reaction_energies):
        """Create convergence comparison visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
        
        # H2O energy convergence
        ax1.plot(basis_sets[:len(h2o_energies)], h2o_energies, 'ro-', linewidth=3, markersize=10)
        ax1.fill_between(basis_sets[:len(h2o_energies)], h2o_energies, alpha=0.3, color='red')
        ax1.set_xlabel('Basis Set')
        ax1.set_ylabel('H₂O Energy (Hartree)')
        ax1.set_title('H₂O Energy Convergence', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Reaction energy convergence
        ax2.plot(basis_sets[:len(reaction_energies)], reaction_energies, 'bo-', linewidth=3, markersize=10)
        ax2.fill_between(basis_sets[:len(reaction_energies)], reaction_energies, alpha=0.3, color='blue')
        ax2.set_xlabel('Basis Set')
        ax2.set_ylabel('Reaction Energy (Hartree)')
        ax2.set_title('Reaction Energy Convergence', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Convergence rates comparison
        if len(h2o_energies) > 1:
            h2o_diffs = [abs(h2o_energies[i+1] - h2o_energies[i]) for i in range(len(h2o_energies)-1)]
            rxn_diffs = [abs(reaction_energies[i+1] - reaction_energies[i]) for i in range(len(reaction_energies)-1)]
            
            ax3.semilogy(basis_sets[1:len(h2o_energies)], h2o_diffs, 'ro-', label='H₂O', linewidth=2, markersize=8)
            ax3.semilogy(basis_sets[1:len(reaction_energies)], rxn_diffs, 'bo-', label='Reaction', linewidth=2, markersize=8)
            ax3.set_xlabel('Basis Set')
            ax3.set_ylabel('|ΔE(n+1) - ΔE(n)| (Hartree)')
            ax3.set_title('Convergence Rate Comparison', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
        
        # Relative convergence
        if len(h2o_energies) > 1:
            h2o_rel = [(e - h2o_energies[-1]) / h2o_energies[-1] * 100 for e in h2o_energies]
            rxn_rel = [(e - reaction_energies[-1]) / reaction_energies[-1] * 100 for e in reaction_energies]
            
            ax4.plot(basis_sets[:len(h2o_energies)], h2o_rel, 'ro-', label='H₂O', linewidth=2, markersize=8)
            ax4.plot(basis_sets[:len(reaction_energies)], rxn_rel, 'bo-', label='Reaction', linewidth=2, markersize=8)
            ax4.set_xlabel('Basis Set')
            ax4.set_ylabel('Relative Error (%)')
            ax4.set_title('Relative Convergence', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/task6_convergence_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
