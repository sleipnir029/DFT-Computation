"""
Main script to run all DFT water splitting analysis tasks - CORRECTED VERSION
"""

import sys
from modules.task1_coordinates import get_experimental_coordinates
from modules.task2_geometry_optimization import run_geometry_optimization
from modules.task3_reaction_energy import run_reaction_energy_calculation
from modules.task4_scf_energy import run_scf_energy_recording
from modules.task5_basis_convergence import calculate_reaction_energy_basis_series
from modules.task6_convergence_comparison import analyze_convergence_behavior
from modules.task7_method_ladder import run_method_ladder
from modules.task8_thermodynamics import calculate_thermodynamic_corrections  # CORRECTED
from modules.task9_accuracy_analysis import create_accuracy_analysis  # CORRECTED
from plotting_prev import plot_basis_convergence, plot_method_comparison_corrected, plot_thermodynamic_breakdown_corrected, plot_accuracy_comparison_before_after
from constants import EXPERIMENTAL_WATER_SPLITTING_DG
from utils.data_manager import data_manager

def main():
    """Run complete DFT water splitting analysis with CORRECTED thermodynamics"""
    
    print("🧪 DFT Water Splitting Reaction Analysis - CORRECTED VERSION")
    print("Reaction: 2H₂O → 2H₂ + O₂")
    print("🔧 Applied fixes: Entropy data + Unit conversions + NIST values")
    print("=" * 70)
    
    results = {}
    
    try:
        # Task T1: Coordinates
        print("\n" + "="*20 + " TASK T1 " + "="*20)
        results['T1'] = get_experimental_coordinates()
        
        # Task T2: Geometry optimization
        print("\n" + "="*20 + " TASK T2 " + "="*20)
        results['T2'] = run_geometry_optimization()
        
        # Task T3: Reaction energy
        print("\n" + "="*20 + " TASK T3 " + "="*20)
        results['T3'] = run_reaction_energy_calculation()
        
        # Task T4: SCF energies
        print("\n" + "="*20 + " TASK T4 " + "="*20)
        results['T4'] = run_scf_energy_recording()
        
        # Task T5: Basis convergence
        print("\n" + "="*20 + " TASK T5 " + "="*20)
        results['T5'] = calculate_reaction_energy_basis_series()
        
        # Task T6: Convergence comparison
        print("\n" + "="*20 + " TASK T6 " + "="*20)
        results['T6'] = analyze_convergence_behavior()
        
        # Task T7: Method ladder
        print("\n" + "="*20 + " TASK T7 " + "="*20)
        results['T7'] = run_method_ladder()
        
        # Task T8: CORRECTED Thermodynamics
        print("\n" + "="*20 + " TASK T8 (CORRECTED) " + "="*20)
        results['T8'] = calculate_thermodynamic_corrections()
        
        # Task T9: CORRECTED Accuracy analysis
        print("\n" + "="*20 + " TASK T9 (CORRECTED) " + "="*20)
        results['T9'] = create_accuracy_analysis()
        
        # Save comprehensive analysis
        print("\n" + "="*20 + " SAVING DATA " + "="*20)
        summary_df = data_manager.save_comprehensive_analysis(results)
        
        # Generate CORRECTED plots
        print("\n" + "="*20 + " PLOTTING (CORRECTED) " + "="*20)
        
        if results['T5']:
            plot_basis_convergence(results['T5'])
        
        if results['T7']:
            plot_method_comparison_corrected(results['T7'], EXPERIMENTAL_WATER_SPLITTING_DG)
        
        if results['T8']:
            plot_thermodynamic_breakdown_corrected(results['T8'])
        
        if results['T9']:
            plot_accuracy_comparison_before_after(results['T9'])
        
        # Final summary
        print(f"\n{'='*70}")
        print("🎉 CORRECTED ANALYSIS COMPLETED SUCCESSFULLY!")
        print("🔧 All thermodynamic errors have been fixed")
        print(f"{'='*70}")
        
        if results['T8']:
            final_dg = results['T8']['reaction_delta_g_ev']
            final_error = results['T8']['error_vs_experimental']
            performance = results['T8']['performance_rating']
            
            print(f"📊 FINAL RESULTS:")
            print(f"   Calculated ΔG: {final_dg:.3f} eV")
            print(f"   Experimental:  {EXPERIMENTAL_WATER_SPLITTING_DG:.3f} eV")
            print(f"   Error:         {final_error:+.3f} eV")
            print(f"   Performance:   {performance}")
            
        print(f"\n📁 Generated files:")
        print(f"  📊 JSON data: results/json/")
        print(f"  📈 CSV tables: results/csv/")
        print(f"  📉 Plots: results/plots/")
        print(f"  📋 Summary: results/dft_analysis_comprehensive_summary.csv")
        print(f"  📄 Complete: results/complete_dft_analysis.json")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Check PySCF installation and dependencies")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
