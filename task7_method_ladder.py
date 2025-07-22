import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pyscf import gto, scf, mp, cc, dft

class Task7MethodLadder:
    def __init__(self, constants):
        self.constants = constants
    
    def run(self, shared_results):
        """T7: Method hierarchy study with proper error handling"""
        print("Task 7: Method hierarchy study")
        
        if 'task2' not in shared_results:
            print("Error: Task 2 must be completed first")
            return None
        
        optimized_geometries = shared_results['task2']['optimized_geometries']
        basis = 'cc-pvdz'
        methods_data = []
        
        # Wave function methods
        wf_methods = ['HF', 'MP2', 'CCSD']
        # DFT methods
        dft_methods = ['PBE', 'PBE0']
        
        for method in wf_methods + dft_methods:
            print(f"Calculating with method: {method}")
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
                    
                    energy = self._calculate_energy(mol, method, mol_name)
                    energies[mol_name] = energy
                
                reaction_energy = (2 * energies['H2'] + energies['O2']) - (2 * energies['H2O'])
                reaction_energy_ev = reaction_energy * self.constants['HARTREE2EV']
                
                methods_data.append({
                    'method': method,
                    'method_type': 'wavefunction' if method in wf_methods else 'dft',
                    'h2_energy': energies['H2'],
                    'o2_energy': energies['O2'],
                    'h2o_energy': energies['H2O'],
                    'reaction_energy_hartree': reaction_energy,
                    'reaction_energy_ev': reaction_energy_ev
                })
                
            except Exception as e:
                print(f"Error with method {method}: {e}")
                continue
        
        # Enhanced visualization
        self._create_method_plots(methods_data)
        
        # Save data
        methods_df = pd.DataFrame(methods_data)
        methods_df.to_csv('csv/task7_method_ladder.csv', index=False)

        with open('json/task7_method_ladder.json', 'w') as f:
            json.dump(methods_data, f, indent=2)
        
        print("Task 7 completed successfully!")
        return methods_data
    
    def _calculate_energy(self, mol, method, mol_name):
        """Calculate energy for given method"""
        if method == 'HF':
            if mol_name == 'O2':
                mf = scf.UHF(mol).run()
            else:
                mf = scf.RHF(mol).run()
            return mf.e_tot
        
        elif method == 'MP2':
            if mol_name == 'O2':
                mf = scf.UHF(mol).run()
            else:
                mf = scf.RHF(mol).run()
            mp2 = mp.MP2(mf).run()
            return mp2.e_tot
        
        elif method == 'CCSD':
            if mol_name == 'O2':
                mf = scf.UHF(mol).run()
            else:
                mf = scf.RHF(mol).run()
            ccsd = cc.CCSD(mf).run()
            return ccsd.e_tot
        
        elif method == 'PBE':
            if mol_name == 'O2':
                mf = dft.UKS(mol)
            else:
                mf = dft.RKS(mol)
            mf.xc = 'PBE'
            mf.run()
            return mf.e_tot
        
        elif method == 'PBE0':
            if mol_name == 'O2':
                mf = dft.UKS(mol)
            else:
                mf = dft.RKS(mol)
            mf.xc = 'PBE0'
            mf.run()
            return mf.e_tot
    
    def _create_method_plots(self, methods_data):
        """Create method comparison visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
        
        methods = [d['method'] for d in methods_data]
        energies = [d['reaction_energy_ev'] for d in methods_data]
        colors = ['red' if d['method_type'] == 'wavefunction' else 'blue' for d in methods_data]
        
        # Main method comparison
        bars = ax1.bar(methods, energies, color=colors, alpha=0.7)
        ax1.axhline(y=4.92, color='green', linestyle='--', linewidth=2, label='Experimental (4.92 eV)')
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Reaction Energy (eV)')
        ax1.set_title('Method Hierarchy: Reaction Energy Comparison', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add energy values on bars
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                    f'{energy:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        # Error from experimental
        experimental_error = [abs(e - 4.92) for e in energies]
        ax2.bar(methods, experimental_error, color=colors, alpha=0.7)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Absolute Error (eV)')
        ax2.set_title('Error from Experimental Value', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Method type comparison
        wf_energies = [d['reaction_energy_ev'] for d in methods_data if d['method_type'] == 'wavefunction']
        dft_energies = [d['reaction_energy_ev'] for d in methods_data if d['method_type'] == 'dft']
        
        ax3.boxplot([wf_energies, dft_energies], labels=['Wavefunction', 'DFT'])
        ax3.axhline(y=4.92, color='green', linestyle='--', linewidth=2, label='Experimental')
        ax3.set_ylabel('Reaction Energy (eV)')
        ax3.set_title('Method Type Comparison', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Jacob's Ladder visualization
        wf_methods_ordered = ['HF', 'MP2', 'CCSD']
        dft_methods_ordered = ['PBE', 'PBE0']
        
        wf_data = [d for d in methods_data if d['method'] in wf_methods_ordered]
        dft_data = [d for d in methods_data if d['method'] in dft_methods_ordered]
        
        if wf_data:
            wf_methods_plot = [d['method'] for d in wf_data]
            wf_energies_plot = [d['reaction_energy_ev'] for d in wf_data]
            ax4.plot(wf_methods_plot, wf_energies_plot, 'ro-', linewidth=2, markersize=8, label='Wavefunction')
        
        if dft_data:
            dft_methods_plot = [d['method'] for d in dft_data]
            dft_energies_plot = [d['reaction_energy_ev'] for d in dft_data]
            ax4.plot(dft_methods_plot, dft_energies_plot, 'bo-', linewidth=2, markersize=8, label='DFT')
        
        ax4.axhline(y=4.92, color='green', linestyle='--', linewidth=2, label='Experimental')
        ax4.set_xlabel('Method')
        ax4.set_ylabel('Reaction Energy (eV)')
        ax4.set_title("Jacob's Ladder", fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/task7_method_ladder.png', dpi=300, bbox_inches='tight')
        plt.show()
