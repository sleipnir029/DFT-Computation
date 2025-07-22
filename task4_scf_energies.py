import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

class Task4SCFEnergies:
    def __init__(self, constants):
        self.constants = constants
    
    def run(self, shared_results):
        """T4: Record SCF energies properly"""
        print("Task 4: Recording SCF total energies")
        
        if 'task3' not in shared_results:
            print("Error: Task 3 must be completed first")
            return None
        
        energies = shared_results['task3']['energies']
        scf_data = []
        
        for mol_name, energy in energies.items():
            scf_data.append({
                'molecule': mol_name,
                'scf_energy_hartree': energy,
                'scf_energy_ev': energy * self.constants['HARTREE2EV'],
                'scf_energy_kj_mol': energy * self.constants['HARTREE2KJ']
            })
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        
        molecules = [d['molecule'] for d in scf_data]
        energies_ev = [d['scf_energy_ev'] for d in scf_data]
        
        colors = ['red', 'blue', 'green']
        bars = ax1.bar(molecules, energies_ev, color=colors, alpha=0.7)
        ax1.set_ylabel('SCF Energy (eV)')
        ax1.set_title('SCF Total Energies', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add energy values
        for bar, energy in zip(bars, energies_ev):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (abs(height) * 0.01),
                    f'{energy:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Energy comparison table
        ax2.axis('tight')
        ax2.axis('off')
        table_data = [[d['molecule'], f"{d['scf_energy_hartree']:.6f}", f"{d['scf_energy_ev']:.1f}"] 
                     for d in scf_data]
        table = ax2.table(cellText=table_data, 
                         colLabels=['Molecule', 'Energy (Hartree)', 'Energy (eV)'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax2.set_title('SCF Energy Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('plots/task4_scf_energies.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save data
        scf_df = pd.DataFrame(scf_data)
        scf_df.to_csv('csv/task4_scf_energies.csv', index=False)

        with open('json/task4_scf_energies.json', 'w') as f:
            json.dump(scf_data, f, indent=2)
        
        print("Task 4 completed successfully!")
        return scf_data
