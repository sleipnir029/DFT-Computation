import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pyscf import gto, scf

class Task3ReactionEnergy:
    def __init__(self, constants):
        self.constants = constants
    
    def run(self, shared_results):
        """T3: Calculate reaction energy with proper spin states"""
        print("Task 3: Calculating reaction energy")
        
        if 'task2' not in shared_results:
            print("Error: Task 2 must be completed first")
            return None
        
        optimized_geometries = shared_results['task2']['optimized_geometries']
        energies = {}
        
        for mol_name, coord in optimized_geometries.items():
            spin = 2 if mol_name == 'O2' else 0
            mol = gto.Mole()
            mol.atom = coord
            mol.basis = 'sto-3g'
            mol.unit = 'Angstrom'
            mol.spin = spin
            mol.build()
            
            if mol_name == 'O2':
                mf = scf.UHF(mol).run()
            else:
                mf = scf.RHF(mol).run()
            
            energies[mol_name] = mf.e_tot
        
        # Reaction energy: 2H2O → 2H2 + O2
        reaction_energy = (2 * energies['H2'] + energies['O2']) - (2 * energies['H2O'])
        reaction_energy_ev = reaction_energy * self.constants['HARTREE2EV']
        
        # Enhanced visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        
        # Energy diagram
        molecules = list(energies.keys())
        molecule_energies = [energies[mol] * self.constants['HARTREE2EV'] for mol in molecules]
        colors = ['red', 'blue', 'green']
        
        bars = ax1.bar(molecules, molecule_energies, color=colors, alpha=0.7)
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title('Molecular Electronic Energies', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add energy values on bars
        for bar, energy in zip(bars, molecule_energies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (height * 0.01),
                    f'{energy:.1f} eV', ha='center', va='bottom', fontweight='bold')
        
        # Reaction energy visualization
        ax2.bar(['Reactants\n(2H2O)', 'Products\n(2H2 + O2)'], 
                [2 * energies['H2O'] * self.constants['HARTREE2EV'], 
                 (2 * energies['H2'] + energies['O2']) * self.constants['HARTREE2EV']],
                color=['red', 'blue'], alpha=0.7, width=0.5)
        
        ax2.annotate('', xy=(1, (2 * energies['H2'] + energies['O2']) * self.constants['HARTREE2EV']), 
                    xytext=(0, 2 * energies['H2O'] * self.constants['HARTREE2EV']),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        
        ax2.text(0.5, (2 * energies['H2O'] * self.constants['HARTREE2EV'] + 
                      (2 * energies['H2'] + energies['O2']) * self.constants['HARTREE2EV']) / 2,
                f'ΔE = {reaction_energy_ev:.3f} eV', ha='center', fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        ax2.set_ylabel('Energy (eV)')
        ax2.set_title('Reaction Energy Diagram', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/task3_reaction_energy.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save data
        reaction_data = pd.DataFrame({
            'molecule': list(energies.keys()),
            'energy_hartree': list(energies.values()),
            'energy_ev': [e * self.constants['HARTREE2EV'] for e in energies.values()]
        })
        reaction_data.to_csv('csv/task3_reaction_energy.csv', index=False)

        with open('json/task3_reaction_energy.json', 'w') as f:
            json.dump({
                'energies': energies,
                'reaction_energy_hartree': reaction_energy,
                'reaction_energy_ev': reaction_energy_ev
            }, f, indent=2)
        
        print(f"Reaction energy: {reaction_energy_ev:.3f} eV")
        print("Task 3 completed successfully!")
        return {
            'energies': energies,
            'reaction_energy_hartree': reaction_energy,
            'reaction_energy_ev': reaction_energy_ev
        }
