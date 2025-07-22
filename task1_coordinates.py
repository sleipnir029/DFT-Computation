import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

class Task1Coordinates:
    def __init__(self, constants):
        self.constants = constants
    
    def run(self, shared_results):
        """T1: Define experimental geometries with precise values"""
        print("Task 1: Defining molecular coordinates")
        
        # More precise experimental geometries from NIST
        geometries = {
            'H2': 'H 0 0 0; H 0 0 0.74168',  # NIST: 0.74168 Å
            'O2': 'O 0 0 0; O 0 0 1.20752',  # NIST: 1.20752 Å  
            'H2O': 'O 0 0 0; H 0.75717 0.58626 0; H -0.75717 0.58626 0'  # NIST geometry
        }
        
        # Enhanced visualization with molecular structures
        fig, axes = plt.subplots(1, 3, figsize=(10, 6))
        
        for i, (mol_name, coord) in enumerate(geometries.items()):
            ax = axes[i]
            if mol_name == 'H2':
                ax.plot([0, 0], [0, 0.74168], 'ro-', markersize=12, linewidth=4, label='H atoms')
                ax.text(0.1, 0.37, 'H-H\n0.742 Å', fontsize=12, fontweight='bold')
                ax.set_ylim(-0.1, 0.85)
            elif mol_name == 'O2':
                ax.plot([0, 0], [0, 1.20752], 'bo-', markersize=12, linewidth=4, label='O atoms')
                ax.text(0.1, 0.6, 'O-O\n1.208 Å', fontsize=12, fontweight='bold')
                ax.set_ylim(-0.1, 1.3)
            elif mol_name == 'H2O':
                # Water molecule geometry
                ax.plot([0, 0.75717], [0, 0.58626], 'ro-', markersize=10, linewidth=3, label='O-H bonds')
                ax.plot([0, -0.75717], [0, 0.58626], 'ro-', markersize=10, linewidth=3)
                ax.plot([0], [0], 'bo', markersize=15, label='O atom')
                # ax.plot([0.75717, -0.75717], [0.58626, 0.58626], 'r', markersize=8)
                ax.text(0.1, 0.3, 'H-O-H\n104.5°', fontsize=12, fontweight='bold')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-0.1, 0.8)
            
            ax.set_title(f'{mol_name} Experimental Geometry', fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('plots/task1_molecular_geometries.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save data
        coord_data = pd.DataFrame({
            'molecule': list(geometries.keys()),
            'coordinates': list(geometries.values())
        })
        coord_data.to_csv('csv/task1_coordinates.csv', index=False)
        
        with open('json/task1_coordinates.json', 'w') as f:
            json.dump(geometries, f, indent=2)
        
        print("Task 1 completed successfully!")
        return geometries
