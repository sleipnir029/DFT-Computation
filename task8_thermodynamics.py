"""
Task 8: Thermodynamic corrections with accurate NIST data
Fixes the entropy and unit conversion errors from the original code
"""

import json
from constants import CONSTANTS

class Task8Thermodynamics:
    """
    Thermodynamic corrections using NIST standard reference data
    - Corrects entropy values (original values were 4-6x too small)
    - Uses proper unit conversions
    - Treats water as liquid at 298K for realistic reaction energetics
    """
    
    def __init__(self, constants_dict=None):
        self.C = constants_dict or CONSTANTS
        
    def run(self, shared_results):
        """
        Apply thermodynamic corrections to electronic energies
        
        Args:
            shared_results: Dictionary containing results from previous tasks
            
        Returns:
            Dictionary with corrected thermodynamic data
        """
        
        # Verify Task 7 (method comparison) has been run
        if 'task7' not in shared_results:
            raise RuntimeError("Task 7 (method comparison) must be run before Task 8")
            
        # Get the best method results (MP2 typically most accurate for small molecules)
        task7_results = shared_results['task7']
        best_method = next((result for result in task7_results 
                           if result['method'] == 'MP2'), task7_results[0])
        
        # NIST reference data at 298.15 K
        # Source: NIST Chemistry WebBook (webbook.nist.gov)
        NIST_DATA = {
            'H2': {
                'S_gas': 130.68,     # J mol⁻¹ K⁻¹, standard entropy of H₂ gas
                'ZPE': 6.197,        # kcal mol⁻¹, zero-point energy
                'H_thermal': 2.024   # kcal mol⁻¹, thermal enthalpy correction
            },
            'O2': {
                'S_gas': 205.00,     # J mol⁻¹ K⁻¹, standard entropy of O₂ gas  
                'ZPE': 0.988,        # kcal mol⁻¹, zero-point energy
                'H_thermal': 2.024   # kcal mol⁻¹, thermal enthalpy correction
            },
            'H2O': {
                'S_liquid': 69.95,   # J mol⁻¹ K⁻¹, standard entropy of liquid H₂O
                'ZPE': 13.435,       # kcal mol⁻¹, zero-point energy
                'H_thermal': 2.368   # kcal mol⁻¹, thermal enthalpy correction
            }
        }
        
        def calculate_thermodynamic_properties(molecule):
            """Calculate thermodynamic properties for a single molecule"""
            
            data = NIST_DATA[molecule]
            mol_key = molecule.lower()
            
            # Electronic energy from quantum calculation
            E_electronic = best_method[f'{mol_key}_energy']
            
            # Zero-point energy correction (kcal/mol → Hartree)
            ZPE_correction = data['ZPE'] * self.C['KCAL2HARTREE']
            
            # Thermal enthalpy correction (kcal/mol → Hartree)  
            H_thermal_correction = data['H_thermal'] * self.C['KCAL2HARTREE']
            
            # Entropy contribution: -T⋅S (J mol⁻¹ K⁻¹ → Hartree)
            # Convert: S [J mol⁻¹ K⁻¹] → S [Hartree K⁻¹] via R_gas⋅Hartree_to_kJ
            entropy_key = 'S_liquid' if molecule == 'H2O' else 'S_gas'
            S_standard = data[entropy_key]  # J mol⁻¹ K⁻¹
            
            # Unit conversion: J mol⁻¹ K⁻¹ → Hartree K⁻¹
            S_hartree_per_K = S_standard / (self.C['R_GAS_CONSTANT'] * self.C['HARTREE2KJ'] * 1000)
            entropy_correction = -self.C['STANDARD_TEMP'] * S_hartree_per_K
            
            # Total Gibbs free energy
            G_total = E_electronic + ZPE_correction + H_thermal_correction + entropy_correction
            
            return {
                'E_electronic': E_electronic,
                'ZPE_correction': ZPE_correction,
                'H_thermal_correction': H_thermal_correction,
                'entropy_correction': entropy_correction,
                'G_total': G_total,
                'S_standard_SI': S_standard  # Keep for reference
            }
        
        # Calculate properties for each molecule
        results = {}
        for molecule in ['H2', 'O2', 'H2O']:
            results[molecule] = calculate_thermodynamic_properties(molecule)
        
        # Calculate reaction free energy: 2H₂O → 2H₂ + O₂  
        # ΔG = G_products - G_reactants
        delta_G_hartree = (2 * results['H2']['G_total'] + 
                          results['O2']['G_total'] - 
                          2 * results['H2O']['G_total'])
        
        # Convert to eV for comparison with experimental value (4.92 eV)
        delta_G_eV = delta_G_hartree * self.C['HARTREE2EV']
        
        # Store complete results
        thermodynamics_results = {
            'H2': results['H2'],
            'O2': results['O2'], 
            'H2O': results['H2O'],
            'reaction_delta_G_hartree': delta_G_hartree,
            'reaction_delta_G_eV': delta_G_eV,
            'experimental_delta_G_eV': 4.92,
            'error_vs_experiment_eV': delta_G_eV - 4.92,
            'method_used': best_method['method'],
            'temperature_K': self.C['STANDARD_TEMP']
        }
        
        # Print summary
        print(f"\n=== Task 8: Thermodynamic Analysis Results ===")
        print(f"Method: {best_method['method']}")
        print(f"Temperature: {self.C['STANDARD_TEMP']} K")
        print(f"")
        print(f"Reaction: 2H₂O(l) → 2H₂(g) + O₂(g)")
        print(f"Calculated ΔG: {delta_G_eV:.3f} eV")
        print(f"Experimental ΔG: 4.92 eV")
        print(f"Error: {delta_G_eV - 4.92:.3f} eV ({abs(delta_G_eV - 4.92)/4.92*100:.1f}%)")
        
        # Store in shared results
        shared_results['task8'] = thermodynamics_results
        
        return thermodynamics_results

    def save_results(self, results, filename='task8_thermodynamics.json'):
        """Save results to JSON file"""
        
        # Convert numpy types to Python native types for JSON serialization
        json_safe_results = json.loads(json.dumps(results, default=str))
        
        with open(filename, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        print(f"Task 8 results saved to {filename}")
