"""
Task 9: Method accuracy analysis with corrected thermodynamics
Analyzes performance of different quantum chemistry methods
"""

import numpy as np
import json
from constants import CONSTANTS

class Task9AccuracyAnalysis:
    """
    Analyze accuracy of different computational methods
    Uses corrected thermodynamic data from Task 8
    """
    
    def __init__(self, constants_dict=None):
        self.C = constants_dict or CONSTANTS
        
    def run(self, shared_results):
        """
        Analyze method accuracy against experimental values
        
        Args:
            shared_results: Dictionary containing all previous task results
            
        Returns:
            Dictionary with accuracy analysis results
        """
        
        if 'task7' not in shared_results or 'task8' not in shared_results:
            raise RuntimeError("Tasks 7 and 8 must be completed before Task 9")
        
        task7_results = shared_results['task7']
        task8_results = shared_results['task8']
        
        # Experimental reference values
        experimental_values = {
            'reaction_delta_G_eV': 4.92,  # eV, water splitting reaction
            'H2O_bond_length': 0.958,     # Å, O-H bond length
            'H2O_bond_angle': 104.5,      # degrees, H-O-H angle
            'H2_bond_length': 0.741,      # Å, H-H bond length
        }
        
        # Analyze each method's performance
        method_analysis = []
        
        for result in task7_results:
            method = result['method']
            
            # Get thermodynamic analysis for this method
            # For simplicity, scale Task 8 results by energy differences
            base_delta_G = task8_results['reaction_delta_G_eV']
            
            # Calculate relative electronic energy differences
            electronic_energy_diff = (
                2 * result['h2_energy'] + result['o2_energy'] - 2 * result['h2o_energy']
            ) * self.C['HARTREE2EV']
            
            # Approximate method-specific thermodynamic correction
            # (In practice, would need full calculation for each method)
            estimated_delta_G = base_delta_G + (electronic_energy_diff - 
                                               (2 * task8_results['H2']['E_electronic'] + 
                                                task8_results['O2']['E_electronic'] - 
                                                2 * task8_results['H2O']['E_electronic']) * 
                                               self.C['HARTREE2EV'])
            
            # Calculate errors
            delta_G_error = estimated_delta_G - experimental_values['reaction_delta_G_eV']
            delta_G_percent_error = abs(delta_G_error) / experimental_values['reaction_delta_G_eV'] * 100
            
            analysis = {
                'method': method,
                'electronic_delta_E_eV': electronic_energy_diff,
                'estimated_delta_G_eV': estimated_delta_G,
                'delta_G_error_eV': delta_G_error,
                'delta_G_percent_error': delta_G_percent_error,
                'abs_delta_G_error_eV': abs(delta_G_error)
            }
            
            # Add geometry errors if available
            if 'geometries' in result:
                geom = result['geometries']
                if 'H2O' in geom:
                    h2o_geom = geom['H2O']
                    if 'bond_length' in h2o_geom:
                        bond_length_error = h2o_geom['bond_length'] - experimental_values['H2O_bond_length']
                        analysis['H2O_bond_length_error_angstrom'] = bond_length_error
                    if 'bond_angle' in h2o_geom:
                        bond_angle_error = h2o_geom['bond_angle'] - experimental_values['H2O_bond_angle']
                        analysis['H2O_bond_angle_error_degrees'] = bond_angle_error
            
            method_analysis.append(analysis)
        
        # Sort by accuracy (smallest absolute error first)
        method_analysis.sort(key=lambda x: x['abs_delta_G_error_eV'])
        
        # Calculate summary statistics
        errors = [abs(m['delta_G_error_eV']) for m in method_analysis]
        summary_stats = {
            'best_method': method_analysis[0]['method'],
            'best_method_error_eV': method_analysis[0]['delta_G_error_eV'],
            'worst_method': method_analysis[-1]['method'],
            'worst_method_error_eV': method_analysis[-1]['delta_G_error_eV'],
            'mean_absolute_error_eV': np.mean(errors),
            'std_error_eV': np.std(errors),
            'methods_within_chemical_accuracy': sum(1 for e in errors if e < 0.1),  # < 0.1 eV
            'chemical_accuracy_percentage': sum(1 for e in errors if e < 0.1) / len(errors) * 100
        }
        
        # Compile final results
        accuracy_results = {
            'method_rankings': method_analysis,
            'summary_statistics': summary_stats,
            'experimental_references': experimental_values,
            'analysis_temperature_K': self.C['STANDARD_TEMP']
        }
        
        # Print results summary
        print(f"\n=== Task 9: Method Accuracy Analysis ===")
        print(f"Best method: {summary_stats['best_method']} "
              f"(error: {summary_stats['best_method_error_eV']:+.3f} eV)")
        print(f"Mean absolute error: {summary_stats['mean_absolute_error_eV']:.3f} eV")
        print(f"Methods within chemical accuracy (±0.1 eV): "
              f"{summary_stats['methods_within_chemical_accuracy']}/{len(method_analysis)} "
              f"({summary_stats['chemical_accuracy_percentage']:.0f}%)")
        
        print(f"\nMethod ranking by ΔG accuracy:")
        for i, method in enumerate(method_analysis[:5], 1):  # Top 5
            print(f"{i}. {method['method']}: {method['delta_G_error_eV']:+.3f} eV")
        
        shared_results['task9'] = accuracy_results
        return accuracy_results
    
    def save_results(self, results, filename='task9_accuracy_analysis.json'):
        """Save results to JSON file"""
        
        json_safe_results = json.loads(json.dumps(results, default=str))
        
        with open(filename, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        print(f"Task 9 results saved to {filename}")
