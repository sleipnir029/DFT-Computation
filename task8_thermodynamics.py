import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

class Task8Thermodynamics:
    def __init__(self, constants):
        self.constants = constants
    
    def run(self, shared_results):
        """T8: Corrected thermodynamic corrections with proper units"""
        print("Task 8: Thermodynamic corrections (CORRECTED)")
        
        if 'task7' not in shared_results:
            print("Error: Task 7 must be completed first")
            return None
        
        # Corrected NIST values at 298.15 K
        nist_data = {
            'H2': {'ZPE': 6.197, 'H_corr': 2.024, 'S': 31.211},
            'O2': {'ZPE': 0.988, 'H_corr': 2.024, 'S': 49.003},
            'H2O': {'ZPE': 13.435, 'H_corr': 2.368, 'S': 45.106}
        }
        
        T = 298.15  # K
        corrections_data = []
        
        # Use best method (MP2 if available, otherwise first method)
        best_method_data = self._get_best_method(shared_results['task7'])
        
        for mol_name in ['H2', 'O2', 'H2O']:
            electronic_energy = best_method_data[f'{mol_name.lower()}_energy']
            
            # Correct unit conversions
            zpe = nist_data[mol_name]['ZPE'] * self.constants['KCAL2HARTREE']
            h_corr = nist_data[mol_name]['H_corr'] * self.constants['KCAL2HARTREE']
            
            # Entropy correction: -T*S (cal/(mol·K) to Hartree)
            s_cal_per_mol_k = nist_data[mol_name]['S']
            s_j_per_mol_k = s_cal_per_mol_k * self.constants['CALORIE2JOULE']
            s_hartree_per_mol_k = s_j_per_mol_k / self.constants['HARTREE2KJ'] * 1000
            entropy_correction = -T * s_hartree_per_mol_k / 1000
            
            # Total Gibbs free energy
            G_298 = electronic_energy + zpe + h_corr + entropy_correction
            
            corrections_data.append({
                'molecule': mol_name,
                'electronic_energy': electronic_energy,
                'zpe_correction': zpe,
                'enthalpy_correction': h_corr,
                'entropy_correction': entropy_correction,
                'gibbs_free_energy': G_298,
                'zpe_kcal_mol': nist_data[mol_name]['ZPE'],
                'entropy_cal_mol_k': s_cal_per_mol_k
            })
        
        # Calculate reaction free energy
        reaction_data = self._calculate_reaction_corrections(corrections_data, best_method_data)
        corrections_data.append(reaction_data)
        
        # Enhanced visualization
        self._create_thermodynamics_plots(corrections_data, best_method_data)
        
        # Save data
        corrections_df = pd.DataFrame(corrections_data)
        corrections_df.to_csv('csv/task8_thermodynamic_corrections.csv', index=False)

        with open('json/task8_thermodynamic_corrections.json', 'w') as f:
            json.dump(corrections_data, f, indent=2)
        
        print("Task 8 completed successfully!")
        return corrections_data
    
    def _get_best_method(self, task7_results):
        """Get best method data (MP2 if available)"""
        for method_data in task7_results:
            if method_data['method'] == 'MP2':
                return method_data
        return task7_results[0]  # Fallback to first method
    
    def _calculate_reaction_corrections(self, corrections_data, best_method_data):
        """Calculate reaction-level corrections"""
        h2_data = next(d for d in corrections_data if d['molecule'] == 'H2')
        o2_data = next(d for d in corrections_data if d['molecule'] == 'O2')
        h2o_data = next(d for d in corrections_data if d['molecule'] == 'H2O')
        
        # Reaction: 2H2O → 2H2 + O2
        reaction_free_energy = (2 * h2_data['gibbs_free_energy'] + 
                              o2_data['gibbs_free_energy']) - (2 * h2o_data['gibbs_free_energy'])
        
        return {
            'molecule': 'Reaction',
            'electronic_energy': best_method_data['reaction_energy_hartree'],
            'zpe_correction': (2*h2_data['zpe_correction'] + 
                             o2_data['zpe_correction'] - 2*h2o_data['zpe_correction']),
            'enthalpy_correction': (2*h2_data['enthalpy_correction'] + 
                                  o2_data['enthalpy_correction'] - 2*h2o_data['enthalpy_correction']),
            'entropy_correction': (2*h2_data['entropy_correction'] + 
                                 o2_data['entropy_correction'] - 2*h2o_data['entropy_correction']),
            'gibbs_free_energy': reaction_free_energy,
            'zpe_kcal_mol': (2*h2_data['zpe_kcal_mol'] + 
                           o2_data['zpe_kcal_mol'] - 2*h2o_data['zpe_kcal_mol']),
            'entropy_cal_mol_k': (2*h2_data['entropy_cal_mol_k'] + 
                                o2_data['entropy_cal_mol_k'] - 2*h2o_data['entropy_cal_mol_k'])
        }
    
    def _create_thermodynamics_plots(self, corrections_data, best_method_data):
        """Create thermodynamics visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
        
        # Correction contributions
        molecules = ['H2', 'O2', 'H2O']
        zpe_vals = [d['zpe_correction'] * self.constants['HARTREE2EV'] 
                   for d in corrections_data if d['molecule'] in molecules]
        h_vals = [d['enthalpy_correction'] * self.constants['HARTREE2EV'] 
                 for d in corrections_data if d['molecule'] in molecules]
        s_vals = [d['entropy_correction'] * self.constants['HARTREE2EV'] 
                 for d in corrections_data if d['molecule'] in molecules]
        
        x = np.arange(len(molecules))
        width = 0.25
        
        ax1.bar(x - width, zpe_vals, width, label='ZPE', color='skyblue', alpha=0.8)
        ax1.bar(x, h_vals, width, label='Enthalpy', color='lightgreen', alpha=0.8)
        ax1.bar(x + width, s_vals, width, label='Entropy', color='salmon', alpha=0.8)
        ax1.set_xlabel('Molecule')
        ax1.set_ylabel('Energy Correction (eV)')
        ax1.set_title('Thermodynamic Corrections by Component', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(molecules)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Electronic vs corrected energies
        electronic_energies = [d['electronic_energy'] * self.constants['HARTREE2EV'] 
                             for d in corrections_data if d['molecule'] in molecules]
        corrected_energies = [d['gibbs_free_energy'] * self.constants['HARTREE2EV'] 
                            for d in corrections_data if d['molecule'] in molecules]
        
        ax2.scatter(electronic_energies, corrected_energies, s=100, alpha=0.7, 
                   c=['red', 'blue', 'green'])
        for i, mol in enumerate(molecules):
            ax2.annotate(mol, (electronic_energies[i], corrected_energies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        min_e = min(min(electronic_energies), min(corrected_energies))
        max_e = max(max(electronic_energies), max(corrected_energies))
        ax2.plot([min_e, max_e], [min_e, max_e], 'r--', alpha=0.7, linewidth=2)
        ax2.set_xlabel('Electronic Energy (eV)')
        ax2.set_ylabel('Corrected Free Energy (eV)')
        ax2.set_title('Electronic vs Thermodynamically Corrected Energies', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Reaction energy comparison
        reaction_correction = next(d for d in corrections_data if d['molecule'] == 'Reaction')
        electronic_rxn = best_method_data['reaction_energy_ev']
        corrected_rxn = reaction_correction['gibbs_free_energy'] * self.constants['HARTREE2EV']
        experimental = 4.92
        
        methods = ['Electronic\n(Best Method)', 'Corrected\n(ΔG₂₉₈)', 'Experimental']
        energies = [electronic_rxn, corrected_rxn, experimental]
        colors = ['lightblue', 'lightgreen', 'gold']
        
        bars = ax3.bar(methods, energies, color=colors, alpha=0.8)
        ax3.set_ylabel('Reaction Energy (eV)')
        ax3.set_title('Water Splitting Reaction Energy Comparison', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{energy:.2f} eV', ha='center', va='bottom', fontweight='bold')
        
        # Error analysis
        error_electronic = abs(electronic_rxn - experimental)
        error_corrected = abs(corrected_rxn - experimental)
        
        ax4.bar(['Electronic', 'Corrected'], [error_electronic, error_corrected], 
                color=['red', 'green'], alpha=0.7)
        ax4.set_ylabel('Absolute Error (eV)')
        ax4.set_title('Error vs Experimental Value', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add improvement percentage
        improvement = (error_electronic - error_corrected) / error_electronic * 100
        ax4.text(0.5, max(error_electronic, error_corrected) * 0.8, 
                f'Improvement: {improvement:.1f}%', ha='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('plots/task8_thermodynamic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"Electronic reaction energy: {electronic_rxn:.3f} eV")
        print(f"Corrected reaction free energy: {corrected_rxn:.3f} eV")
        print(f"Experimental value: {experimental:.3f} eV")
        print(f"Improvement: {error_electronic:.3f} → {error_corrected:.3f} eV")
        print(f"ZPE correction magnitude: {abs(reaction_correction['zpe_correction'] * self.constants['HARTREE2EV']):.3f} eV")
        print(f"Entropy correction magnitude: {abs(reaction_correction['entropy_correction'] * self.constants['HARTREE2EV']):.3f} eV")
