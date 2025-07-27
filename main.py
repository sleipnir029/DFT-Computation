"""
Main script to run all DFT water splitting analysis tasks
Professional scientific analysis with comprehensive plotting
"""

import sys
from pkg_resources import resource_string
from modules.task1_coordinates import get_experimental_coordinates
from modules.task2_geometry_optimization import run_geometry_optimization
from modules.task3_reaction_energy import run_reaction_energy_calculation
from modules.task4_scf_energy import run_scf_energy_recording
from modules.task5_basis_convergence import calculate_reaction_energy_basis_series
from modules.task6_convergence_comparison import analyze_convergence_behavior
from modules.task7_method_ladder import run_method_ladder
from modules.task8_thermodynamics import calculate_thermodynamic_corrections
from modules.task9_accuracy_analysis import create_accuracy_analysis

# Updated plotting import - uses new scientific plotting module
from plotting import generate_all_plots

from constants import EXPERIMENTAL_WATER_SPLITTING_DG
from data_manager import data_manager

def main():
    """Run complete DFT water splitting analysis with professional scientific plots"""
    
    print("🧪 DFT Water Splitting Reaction Analysis")
    print("Reaction: 2H₂O → 2H₂ + O₂")
    print("🔬 Professional Scientific Analysis with Publication-Quality Plots")
    print("=" * 70)
    
    results = {}
    
    try:
        # Task T1: Experimental Coordinates
        print("\n" + "="*20 + " TASK T1 " + "="*20)
        results['T1'] = get_experimental_coordinates()
        
        # Task T2: Geometry Optimization
        print("\n" + "="*20 + " TASK T2 " + "="*20)
        results['T2'] = run_geometry_optimization()
        
        # Task T3: Reaction Energy Calculation
        print("\n" + "="*20 + " TASK T3 " + "="*20)
        results['T3'] = run_reaction_energy_calculation()
        
        # Task T4: SCF Energy Recording
        print("\n" + "="*20 + " TASK T4 " + "="*20)
        results['T4'] = run_scf_energy_recording()
        
        # Task T5: Basis Set Convergence
        print("\n" + "="*20 + " TASK T5 " + "="*20)
        results['T5'] = calculate_reaction_energy_basis_series()
        
        # Task T6: Convergence Comparison
        print("\n" + "="*20 + " TASK T6 " + "="*20)
        results['T6'] = analyze_convergence_behavior()
        
        # Task T7: Method Hierarchy (Jacob's Ladder)
        print("\n" + "="*20 + " TASK T7 " + "="*20)
        results['T7'] = run_method_ladder()
        
        # Task T8: Thermodynamic Corrections
        print("\n" + "="*20 + " TASK T8 " + "="*20)
        results['T8'] = calculate_thermodynamic_corrections()
        
        # Task T9: Accuracy Analysis
        print("\n" + "="*20 + " TASK T9 " + "="*20)
        results['T9'] = create_accuracy_analysis()
        
        # Save comprehensive analysis data
        print("\n" + "="*20 + " SAVING DATA " + "="*20)
        summary_df = data_manager.save_comprehensive_analysis(results)
        
        # Generate all scientific plots for research report
        print("\n" + "="*20 + " GENERATING SCIENTIFIC PLOTS " + "="*20)
        figures = generate_all_plots(results)

        # Display final results summary
        print(f"\n{'='*70}")
        print("🎉 DFT ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        
        # Show key results
        if results.get('T8') and isinstance(results['T8'], dict):
            final_dg = results['T8'].get('reaction_delta_g_ev', 0.0)
            final_error = results['T8'].get('error_vs_experimental', 0.0)
            performance = results['T8'].get('performance_rating', 'Unknown')
            
            print(f"\n📊 FINAL SCIENTIFIC RESULTS:")
            print(f"   Reaction: 2H₂O(l) → 2H₂(g) + O₂(g)")
            print(f"   Calculated ΔG: {final_dg:.3f} eV")
            print(f"   Experimental:  {EXPERIMENTAL_WATER_SPLITTING_DG:.3f} eV")
            print(f"   Error:         {final_error:+.3f} eV ({abs(final_error)/EXPERIMENTAL_WATER_SPLITTING_DG*100:.1f}%)")
            print(f"   Performance:   {performance}")
            
        # Show best method
        if results.get('T9') is not None and isinstance(results['T9'], dict) and 'best_method' in results['T9']:
            best = results['T9']['best_method']
            print(f"\n🏆 RECOMMENDED METHOD: {best['method']}")
            print(f"   Final accuracy: {best['error']:+.3f} eV")
            
        # Show accuracy statistics
        if results.get('T9') and isinstance(results['T9'], dict):
            chem_acc = results['T9'].get('chemical_accurate_count', 0)
            exc_acc = results['T9'].get('excellent_accurate_count', 0)
            total_methods = len(results['T9'].get('analysis_data', []))
            
            if total_methods > 0:
                print(f"\n📈 ACCURACY STATISTICS:")
                print(f"   Chemical accuracy (±1 kcal/mol): {chem_acc}/{total_methods} methods ({chem_acc/total_methods*100:.0f}%)")
                print(f"   Excellent accuracy (±0.1 eV):    {exc_acc}/{total_methods} methods ({exc_acc/total_methods*100:.0f}%)")
        
        # File outputs summary
        print(f"\n📁 GENERATED OUTPUT FILES:")
        print(f"   📊 Raw Data (JSON):     results/json/")
        print(f"   📈 Analysis Tables:     results/csv/")
        print(f"   📉 Scientific Plots:    results/plots/")
        print(f"   📋 Summary Report:      results/dft_analysis_comprehensive_summary.csv")
        print(f"   📄 Complete Dataset:    results/complete_dft_analysis.json")
        
        print(f"\n🔬 PUBLICATION-READY SCIENTIFIC PLOTS GENERATED:")
        plot_files = [
            "basis_set_convergence.png - Basis set convergence analysis", 
            "method_performance_analysis.png - Method comparison & Jacob's ladder",
            "thermodynamic_contributions.png - Thermodynamic breakdown analysis",
            "convergence_analysis.png - Error cancellation demonstration", 
            "comprehensive_summary.png - Complete results overview"
        ]
        for plot_file in plot_files:
            print(f"   📊 {plot_file}")
        
        print(f"\n✅ Analysis ready for research report writing!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("💡 Troubleshooting steps:")
        print("   1. Check PySCF installation: pip install pyscf")
        print("   2. Verify all task files are present")
        print("   3. Ensure data_manager.py and constants.py are available")
        
        import traceback
        traceback.print_exc()
        return None

def print_analysis_info():
    """Print information about the DFT analysis"""
    
    print("\n" + "="*70)
    print("DFT WATER SPLITTING ANALYSIS - SCIENTIFIC COMPUTING PROJECT")
    print("="*70)
    print("🔬 COMPUTATIONAL DETAILS:")
    print("   • Quantum Chemistry Methods: HF, MP2, CCSD, PBE, PBE0")
    print("   • Basis Sets: STO-3G → cc-pV5Z convergence study")
    print("   • Thermodynamic Corrections: ZPE + Thermal + Entropy (298.15K)")
    print("   • Reference Data: NIST Chemistry WebBook")
    print("")
    print("📊 ANALYSIS TASKS:")
    tasks = [
        "T1: Experimental molecular coordinates",
        "T2: Geometry optimization (HF/STO-3G)", 
        "T3: Electronic reaction energy calculation",
        "T4: SCF total energy recording",
        "T5: Basis set convergence study",
        "T6: Convergence behavior analysis", 
        "T7: Method hierarchy (Jacob's ladder)",
        "T8: Thermodynamic corrections (NIST data)",
        "T9: Final accuracy assessment"
    ]
    for task in tasks:
        print(f"   • {task}")
    
    print("\n🎯 TARGET ACCURACY: Chemical accuracy (±1 kcal/mol = ±0.043 eV)")
    print("📚 EXPERIMENTAL REFERENCE: ΔG = 4.92 eV (water splitting)")
    print("="*70)

if __name__ == "__main__":
    # Print analysis information
    print_analysis_info()
    
    # Run the complete analysis
    results = main()
    
    # Final status
    if results:
        print(f"\n🎉 SUCCESS: DFT water splitting analysis completed!")
        print(f"📄 Ready for scientific report writing and publication.")
    else:
        print(f"\n❌ FAILED: Analysis could not be completed.")
        print(f"🔧 Please check error messages above and fix issues.")
