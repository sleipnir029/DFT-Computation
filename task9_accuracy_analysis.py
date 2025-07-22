import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

class Task9AccuracyAnalysis:
    def __init__(self, constants):
        self.constants = constants
    
    def run(self, shared_results):
        """T9: Comprehensive accuracy analysis"""
        print("Task 9: Accuracy analysis")
        
        if 'task8' not in shared_results or 'task7' not in shared_results:
            print("Error: Tasks 7 and 8 must be completed first")
            return None
        
        experimental_dg = 4.92  # eV
        
        # Get thermodynamic corrections
        thermo_corrections = shared_results['task8']
        reaction_correction = next(d for d in thermo_corrections if d['molecule'] == 'Reaction')
        
        # Analyze all methods
        analysis_data = []
        
        for method_data in shared_results['task7']:
            method = method_data['method']
            electronic_energy = method_data['reaction_energy_ev']
            
            # Apply proper thermodynamic corrections
            zpe_corr = reaction_correction['zpe_correction'] * self.constants['HARTREE2EV']
            h_corr = reaction_correction['enthalpy_correction'] * self.constants['HARTREE2EV']
            s_corr = reaction_correction['entropy_correction'] * self.constants['HARTREE2EV']
            
            corrected_energy = electronic_energy + zpe_corr + h_corr + s_corr
            deviation = abs(corrected_energy - experimental_dg)
            
            analysis_data.append({
                'method': method,
                'method_type': method_data['method_type'],
                'electronic_energy': electronic_energy,
                'zpe_correction': zpe_corr,
                'enthalpy_correction': h_corr,
                'entropy_correction': s_corr,
                'corrected_energy': corrected_energy,
                'experimental': experimental_dg,
                'deviation': deviation,
                'percent_error': (deviation / experimental_dg) * 100
            })
        
        # Enhanced visualization
        self._create_accuracy_plots(analysis_data, experimental_dg)
        
        # Print summary
        self._print_accuracy_summary(analysis_data, experimental_dg)
        
        # Save data
        analysis_df = pd.DataFrame(analysis_data)
        analysis_df.to_csv('csv/task9_accuracy_analysis.csv', index=False)

        with open('json/task9_accuracy_analysis.json', 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print("Task 9 completed successfully!")
        return analysis_data
    
    def _create_accuracy_plots(self, analysis_data, experimental_dg):
        """Create comprehensive accuracy visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
        
        methods = [d['method'] for d in analysis_data]
        electronic_energies = [d['electronic_energy'] for d in analysis_data]
        corrected_energies = [d['corrected_energy'] for d in analysis_data]
        deviations = [d['deviation'] for d in analysis_data]
        
        # Method comparison with corrections
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, electronic_energies, width, label='Electronic', alpha=0.7)
        bars2 = ax1.bar(x + width/2, corrected_energies, width, label='Corrected', alpha=0.7)
        ax1.axhline(y=experimental_dg, color='red', linestyle='--', linewidth=2, 
                   label=f'Experimental ({experimental_dg} eV)')
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Reaction Energy (eV)')
        ax1.set_title('Electronic vs Corrected Reaction Energies', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Deviation from experimental
        colors = ['red' if d['method_type'] == 'wavefunction' else 'blue' for d in analysis_data]
        bars = ax2.bar(methods, deviations, color=colors, alpha=0.7)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Deviation from Experiment (eV)')
        ax2.set_title('Accuracy Analysis: Deviation from Experimental Value', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add deviation values on bars
        for bar, deviation in zip(bars, deviations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{deviation:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Thermodynamic corrections breakdown
        corrections = ['ZPE', 'Enthalpy', 'Entropy']
        avg_corrections = [
            np.mean([d['zpe_correction'] for d in analysis_data]),
            np.mean([d['enthalpy_correction'] for d in analysis_data]),
            np.mean([d['entropy_correction'] for d in analysis_data])
        ]
        
        ax3.bar(corrections, avg_corrections, color=['skyblue', 'lightgreen', 'salmon'], alpha=0.7)
        ax3.set_ylabel('Average Correction (eV)')
        ax3.set_title('Thermodynamic Corrections Breakdown', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add correction values on bars
        for i, (corr, val) in enumerate(zip(corrections, avg_corrections)):
            ax3.text(i, val + (0.01 if val > 0 else -0.01), f'{val:.3f}', 
                    ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
        
        # Method performance ranking
        sorted_data = sorted(analysis_data, key=lambda x: x['deviation'])
        sorted_methods = [d['method'] for d in sorted_data]
        sorted_deviations = [d['deviation'] for d in sorted_data]
        sorted_colors = ['red' if d['method_type'] == 'wavefunction' else 'blue' for d in sorted_data]
        
        ax4.barh(sorted_methods, sorted_deviations, color=sorted_colors, alpha=0.7)
        ax4.set_xlabel('Deviation from Experiment (eV)')
        ax4.set_ylabel('Method (Ranked by Accuracy)')
        ax4.set_title('Method Performance Ranking', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add "Best" and "Worst" labels
        if len(sorted_methods) > 1:
            ax4.text(sorted_deviations[0] + 0.1, 0, 'Best', va='center', fontweight='bold', color='green')
            ax4.text(sorted_deviations[-1] + 0.1, len(sorted_methods)-1, 'Worst', va='center', fontweight='bold', color='red')
        
        plt.tight_layout()
        plt.savefig('plots/task9_accuracy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _print_accuracy_summary(self, analysis_data, experimental_dg):
        """Print detailed accuracy summary"""
        sorted_data = sorted(analysis_data, key=lambda x: x['deviation'])
        avg_corrections = [
            np.mean([d['zpe_correction'] for d in analysis_data]),
            np.mean([d['enthalpy_correction'] for d in analysis_data]),
            np.mean([d['entropy_correction'] for d in analysis_data])
        ]
        
        print("\n" + "="*50)
        print("ACCURACY ANALYSIS SUMMARY")
        print("="*50)
        print(f"Experimental value: {experimental_dg:.2f} eV")
        print(f"Best method: {sorted_data[0]['method']} (deviation: {sorted_data[0]['deviation']:.3f} eV)")
        print(f"Worst method: {sorted_data[-1]['method']} (deviation: {sorted_data[-1]['deviation']:.3f} eV)")
        print(f"Average ZPE correction: {avg_corrections[0]:.3f} eV")
        print(f"Average enthalpy correction: {avg_corrections[1]:.3f} eV")
        print(f"Average entropy correction: {avg_corrections[2]:.3f} eV")
        print("\nMethod Ranking (by accuracy):")
        for i, data in enumerate(sorted_data, 1):
            print(f"{i}. {data['method']}: {data['deviation']:.3f} eV deviation")
        print("="*50)
