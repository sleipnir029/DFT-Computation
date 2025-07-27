"""
Scientific plotting utilities for DFT water splitting analysis
Professional plots for research report - no before/after comparisons

Usage:
    from dft_analysis import plotting
    # results = ... (load or compute your results dictionary)
    plotting.generate_all_plots(results)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(sns.color_palette("husl"))

# Create plots directory
plots_dir = Path("results/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# Configure matplotlib for publication quality
# plt.rcParams.update({
#     'font.size': 12,
#     'font.family': 'DejaVu Sans',
#     'axes.linewidth': 1.2,
#     'grid.alpha': 0.3,
#     'lines.linewidth': 2,
#     'lines.markersize': 8,
#     'legend.frameon': True,
#     'legend.fancybox': False,
#     'legend.shadow': True,
#     'figure.dpi': 300,
#     'savefig.dpi': 300,
#     'savefig.bbox': 'tight'
# })

plt.rcParams.update({
    'font.size': 11,                  # Slightly smaller, cleaner text
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 0.8,            # Thinner axis lines for clarity
    'grid.alpha': 0.15,               # Softer grid (less distracting)
    'lines.linewidth': 1.5,           # Slimmer lines
    'lines.markersize': 6,            # Smaller markers (reduce crowding)
    'legend.frameon': True,          # Remove legend frames (cleaner)
    'legend.fancybox': False,
    'legend.shadow': False,           # No shadows (reduces visual noise)
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.titlesize': 12,             # Consistent, slightly reduced title size
    'axes.labelsize': 11,             # Reduce axis label size
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,                # Keep grid but make it subtle
    'axes.grid.axis': 'y',            # Only horizontal gridlines (less clutter)
})


def plot_basis_set_convergence(basis_results, save=True):
    """
    Plot basis set convergence for water splitting reaction
    Shows how reaction energy converges with increasing basis set size
    """
    basis_names = []
    energies = []
    basis_sizes = []  # Approximate basis function counts
    
    # Basis set size mapping (approximate number of functions for H2O)
    size_map = {
        'STO-3G': 7, '3-21G': 13, '6-31G': 13, 'cc-pVDZ': 24, 
        'cc-pVTZ': 58, 'cc-pVQZ': 115, 'cc-pV5Z': 201
    }
    
    for basis, result in basis_results.items():
        if result is not None:
            basis_names.append(basis)
            energies.append(result['reaction_energy_ev'])
            basis_sizes.append(size_map.get(basis, 50))
    
    if not energies:
        print("No valid basis set data for plotting")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot 1: Energy vs Basis Set
    ax1.plot(range(len(basis_names)), energies, 'o-', linewidth=2.5, 
             markersize=8, color='#2E86AB', markerfacecolor='white', 
             markeredgewidth=2, markeredgecolor='#2E86AB')
    
    ax1.set_ylim(1.3, 5.1)  # Set y-limits to focus on reaction energy range
    ax1.set_xlabel('Basis Set', fontweight='bold')
    ax1.set_ylabel('Reaction Energy (eV)', fontweight='bold')
    ax1.set_title('Basis Set Convergence: 2H₂O → 2H₂ + O₂', fontweight='bold', fontsize=14)
    ax1.set_xticks(range(len(basis_names)))
    ax1.set_xticklabels(basis_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add experimental reference line
    ax1.axhline(y=4.92, color='red', linestyle='--', linewidth=2, 
                alpha=0.8, label='Experimental (4.92 eV)')
    ax1.legend()
    
    # Add energy values as annotations
    for i, (basis, energy) in enumerate(zip(basis_names, energies)):
        ax1.annotate(f'{energy:.2f}', (i, energy), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # # Plot 2: Energy vs Basis Size (shows convergence trend)
    # ax2.scatter(basis_sizes, energies, s=120, c='#A23B72', alpha=0.7, edgecolors='black')
    
    # Plot 2: No quadratic fit, just connect points
    ax2.plot(basis_sizes, energies, 'o-', color='#A23B72', linewidth=2.5,
            markersize=8, markerfacecolor='white', markeredgewidth=2, markeredgecolor='#A23B72')
    ax2.set_ylim(1.3, 5.1)  # Set y-limits to focus on reaction energy range
    # # Fit trend line
    # if len(basis_sizes) > 2:
    #     z = np.polyfit(basis_sizes, energies, 2)  # Quadratic fit
    #     p = np.poly1d(z)
    #     x_trend = np.linspace(min(basis_sizes), max(basis_sizes)*1.1, 100)
    #     ax2.plot(x_trend, p(x_trend), '--', color='gray', alpha=0.8, label='Trend')
    
    ax2.set_xlabel('Approximate Basis Functions', fontweight='bold')
    ax2.set_ylabel('Reaction Energy (eV)', fontweight='bold') 
    ax2.set_title('Convergence with Basis Set Size', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add basis set labels
    for basis, size, energy in zip(basis_names, basis_sizes, energies):
        ax2.annotate(basis, (size, energy), textcoords="offset points", 
                    xytext=(5,5), ha='left', fontsize=9, alpha=0.8)
    
    # # Plot 2: No quadratic fit, just connect points
    # ax2.plot(basis_sizes, energies, 'o-', color='#A23B72', linewidth=1.5,
    #         markersize=5, markerfacecolor='white', markeredgewidth=1, markeredgecolor='#A23B72')

    # # Reduce clutter
    # ax2.tick_params(labelsize=9)
    # for basis, size, energy in zip(basis_names, basis_sizes, energies):
    #     ax2.annotate(basis, (size, energy), xytext=(3,3), ha='left', fontsize=8, alpha=0.7)

    
    plt.tight_layout()
    
    if save:
        save_path = plots_dir / 'basis_set_convergence.png'
        plt.savefig(save_path)
        print(f"✅ Saved: {save_path}")
    
    plt.show()
    return fig

def plot_method_performance(method_results, experimental_value=4.92, save=True):
    """
    Plot quantum chemistry method performance analysis
    Shows accuracy and method hierarchy (Jacob's ladder)
    """
    
    # Extract and organize data
    methods = []
    energies = []
    errors = []
    method_types = []
    
    # Method categorization for Jacob's ladder
    wf_methods = ['HF', 'MP2', 'CCSD', 'CCSD(T)']
    dft_gga = ['PBE', 'BLYP', 'BP86']
    dft_hybrid = ['PBE0', 'B3LYP', 'M06-2X']
    
    for method, result in method_results.items():
        if result is not None:
            methods.append(method)
            energies.append(result['reaction_energy_ev'])
            errors.append(result['reaction_energy_ev'] - experimental_value)
            
            if method in wf_methods:
                method_types.append('Wavefunction')
            elif method in dft_gga:
                method_types.append('DFT-GGA')
            elif method in dft_hybrid:
                method_types.append('DFT-Hybrid')
            else:
                method_types.append('Other')
    
    if not methods:
        print("No valid method data for plotting")
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
    
    # Plot 1: Method accuracy bar chart
    colors = ['#1f77b4' if t == 'Wavefunction' else '#ff7f0e' if t == 'DFT-GGA' 
              else '#2ca02c' if t == 'DFT-Hybrid' else '#d62728' for t in method_types]
    
    bars = ax1.bar(methods, energies, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_ylim(0, 6)
    ax1.axhline(y=experimental_value, color='red', linestyle='--', linewidth=2,
                label=f'Experimental ({experimental_value:.2f} eV)')
    ax1.set_ylabel('Reaction Energy (eV)', fontweight='bold')
    ax1.set_title('Method Comparison: Water Splitting Energies', fontweight='bold', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add energy values on bars
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{energy:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Error analysis
    error_colors = ['green' if abs(e) < 0.1 else 'orange' if abs(e) < 0.3 else 'red' for e in errors]
    bars2 = ax2.bar(methods, errors, color=error_colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_ylim(-0.9, 0.1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=0.043, color='green', linestyle=':', alpha=0.7, label='Chemical accuracy (±1 kcal/mol)')
    ax2.axhline(y=-0.043, color='green', linestyle=':', alpha=0.7)
    ax2.set_ylabel('Error vs Experiment (eV)', fontweight='bold')
    ax2.set_title('Method Accuracy Assessment', fontweight='bold', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add error values
    for bar, error in zip(bars2, errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.02,
                f'{error:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

    # Plot 3: Jacob's ladder representation
    ladder_order = []
    ladder_energies = []
    ladder_colors = []
    
    # Order methods by theoretical hierarchy
    method_order = ['HF', 'MP2', 'CCSD', 'PBE', 'PBE0']
    for method in method_order:
        if method in methods:
            idx = methods.index(method)
            ladder_order.append(method)
            ladder_energies.append(energies[idx])
            ladder_colors.append(colors[idx])
    
    if ladder_order:
        ax3.plot(range(len(ladder_order)), ladder_energies, 'o-', linewidth=2.5, markersize=10)
        ax3.set_ylim(4, 5)
        ax3.axhline(y=experimental_value, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Method Hierarchy', fontweight='bold')
        ax3.set_ylabel('Reaction Energy (eV)', fontweight='bold')
        ax3.set_title("Jacob's Ladder: Method Hierarchy", fontweight='bold', fontsize=14)
        ax3.set_xticks(range(len(ladder_order)))
        ax3.set_xticklabels(ladder_order)
        ax3.grid(True, alpha=0.3)
        
        # Annotate method types
        for i, (method, energy) in enumerate(zip(ladder_order, ladder_energies)):
            method_type = 'WF' if method in wf_methods else 'GGA' if method in dft_gga else 'Hybrid'
            ax3.annotate(method_type, (i, energy), textcoords="offset points", 
                        xytext=(0,-10), ha='center', va="top", fontsize=9, alpha=0.7)

    # Plot 4: Performance statistics
    abs_errors = [abs(e) for e in errors]
    chemical_acc = sum(1 for e in abs_errors if e < 0.043)
    excellent_acc = sum(1 for e in abs_errors if e < 0.1)
    good_acc = sum(1 for e in abs_errors if e < 0.3)
    
    categories = ['Chemical\n(±1 kcal/mol)', 'Excellent\n(±0.1 eV)', 'Good\n(±0.3 eV)', 'Total Methods']
    counts = [chemical_acc, excellent_acc, good_acc, len(methods)]
    percentages = [c/len(methods)*100 for c in counts[:-1]] + [100]
    
    bars4 = ax4.bar(categories, counts, color=['green', 'blue', 'orange', 'gray'], alpha=0.7)
    ax4.set_ylim(0, max(counts) + 1)
    ax4.set_ylabel('Number of Methods', fontweight='bold')
    ax4.set_title('Accuracy Distribution', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, count, pct in zip(bars4, counts, percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{count}\n({pct:.0f}%)', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        save_path = plots_dir / 'method_performance_analysis.png'
        plt.savefig(save_path)
        print(f"✅ Saved: {save_path}")
    
    plt.show()
    return fig

def plot_thermodynamic_contributions(thermo_results, save=True):
    """
    Plot thermodynamic contributions breakdown
    Shows ZPE, thermal, and entropy corrections for each molecule
    """
    
    if not thermo_results or 'corrected_energies' not in thermo_results:
        print("No thermodynamic data available for plotting")
        return None
    
    molecules = ['H₂', 'O₂', 'H₂O']
    mol_keys = ['H2', 'O2', 'H2O']
    
    # Extract data
    electronic_data = []
    zpe_data = []
    thermal_data = []
    entropy_data = []
    total_data = []
    
    for mol_key in mol_keys:
        if mol_key in thermo_results['corrected_energies']:
            data = thermo_results['corrected_energies'][mol_key]
            electronic_data.append(data['electronic'] * 27.211)  # Convert to eV
            zpe_data.append(data['zpe'] * 27.211)
            thermal_data.append(data['thermal'] * 27.211)
            entropy_data.append(data['entropy'] * 27.211)
            total_data.append(data['total_g'] * 27.211)
    
    if not electronic_data:
        print("No valid thermodynamic data for plotting")
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))

    # Plot 1: Stacked thermodynamic contributions
    x = np.arange(len(molecules))
    width = 0.6
    
    p1 = ax1.bar(x, zpe_data, width, label='Zero-Point Energy', color='#1f77b4', alpha=0.8)
    p2 = ax1.bar(x, thermal_data, width, bottom=zpe_data, label='Thermal Enthalpy', color='#ff7f0e', alpha=0.8)
    
    # For entropy (can be negative), handle separately
    entropy_bottoms = [z + t for z, t in zip(zpe_data, thermal_data)]
    p3 = ax1.bar(x, entropy_data, width, bottom=entropy_bottoms, label='-T·S (Entropy)', color='#2ca02c', alpha=0.8)
    
    ax1.set_xlabel('Molecule', fontweight='bold')
    ax1.set_ylabel('Energy Contribution (eV)', fontweight='bold')
    ax1.set_title('Thermodynamic Corrections at 298.15 K', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(molecules)
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (zpe, thermal, entropy) in enumerate(zip(zpe_data, thermal_data, entropy_data)):
        ax1.text(i, zpe/2, f'{zpe:.3f}', ha='center', va='top', fontweight='bold', color='white')
        ax1.text(i, zpe + thermal/2, f'{thermal:.3f}', ha='center', va='center', fontweight='bold', color='white')
        ax1.text(i, zpe + thermal + entropy/2, f'{entropy:.3f}', ha='center', va='bottom', fontweight='bold', color='white')
    
    # Plot 2: Electronic vs Total energies
    x_pos = np.arange(len(molecules))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, electronic_data, width, label='Electronic Energy', 
                    color='#d62728', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, total_data, width, label='Total Free Energy (G)', 
                    color='#9467bd', alpha=0.7)
    
    ax2.set_xlabel('Molecule', fontweight='bold')
    ax2.set_ylabel('Energy (eV)', fontweight='bold')
    ax2.set_title('Electronic vs Thermodynamically Corrected Energies', fontweight='bold', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(molecules)
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Contribution magnitudes
    contributions = ['ZPE', 'Thermal H', '|T·S|']
    avg_contributions = [
        np.mean([abs(x) for x in zpe_data]),
        np.mean([abs(x) for x in thermal_data]),
        np.mean([abs(x) for x in entropy_data])
    ]
    
    bars3 = ax3.bar(contributions, avg_contributions, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax3.set_ylim(0, max(avg_contributions) + 0.1)
    ax3.set_ylabel('Average Contribution (eV)', fontweight='bold')
    ax3.set_title('Average Thermodynamic Correction Magnitudes', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    for bar, value in zip(bars3, avg_contributions):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Reaction energy breakdown
    if 'reaction_delta_g_ev' in thermo_results:
        electronic_delta = thermo_results['electronic_results']['reaction_energy_ev']
        thermodynamic_delta = thermo_results['reaction_delta_g_ev']
        experimental = 4.92
        
        components = ['Electronic\nΔE', 'Thermodynamic\nΔG', 'Experimental\nΔG']
        values = [electronic_delta, thermodynamic_delta, experimental]
        colors = ['#d62728', '#9467bd', 'red']
        
        bars4 = ax4.bar(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax4.set_ylim(0, max(values) + 0.5)
        ax4.axhline(y=experimental, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax4.set_ylabel('Reaction Energy (eV)', fontweight='bold')
        ax4.set_title('Water Splitting Reaction: Electronic vs Thermodynamic', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add error annotation above the tallest bar
        # This annotation box summarizes the deviation from experiment for clarity
        error = thermodynamic_delta - experimental
        max_height = max(values)
        ax4.text(1, max_height - 1, f'Error: {error:+.3f} eV ({abs(error)/experimental*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    
    if save:
        save_path = plots_dir / 'thermodynamic_contributions.png'
        plt.savefig(save_path)
        print(f"✅ Saved: {save_path}")
    
    plt.show()
    return fig

def plot_convergence_analysis(convergence_results, save=True):
    """
    Plot convergence behavior comparison.
    Shows H2O vs reaction energy convergence patterns.
    The second subplot uses a logarithmic scale for the y-axis to highlight small differences in convergence errors.
    """
    
    if not convergence_results:
        print("No convergence data available for plotting")
        return None
    
    basis_sets = convergence_results.get('basis_sets', [])
    h2o_convergence = convergence_results.get('h2o_convergence', [])
    reaction_convergence = convergence_results.get('reaction_convergence', [])
    
    if not basis_sets:
        print("No valid convergence data for plotting")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot 1: Convergence trajectories
    x = range(len(basis_sets))
    
    ax1.plot(x, h2o_convergence, 'o-', linewidth=2.5, markersize=8, 
             color='#1f77b4', label='H₂O Total Energy', markerfacecolor='white',
             markeredgewidth=2, markeredgecolor='#1f77b4')
    ax1.plot(x, reaction_convergence, 's-', linewidth=2.5, markersize=8,
             color='#ff7f0e', label='Reaction Energy', markerfacecolor='white',
             markeredgewidth=2, markeredgecolor='#ff7f0e')
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Basis Set', fontweight='bold')
    ax1.set_ylabel('Convergence Error (eV)', fontweight='bold')
    ax1.set_title('Convergence Behavior: H₂O vs Reaction Energy', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(basis_sets, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add convergence threshold lines
    ax1.axhline(y=0.043, color='green', linestyle=':', alpha=0.7, label='Chemical accuracy')
    ax1.axhline(y=-0.043, color='green', linestyle=':', alpha=0.7)
    
    # Plot 2: Absolute convergence comparison
    abs_h2o = [abs(x) for x in h2o_convergence]
    abs_reaction = [abs(x) for x in reaction_convergence]
    
    x_pos = np.arange(len(basis_sets))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, abs_h2o, width, label='|H₂O Convergence|', 
                    color='#1f77b4', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, abs_reaction, width, label='|Reaction Convergence|', 
                    color='#ff7f0e', alpha=0.7)
    
    ax2.set_xlabel('Basis Set', fontweight='bold')
    ax2.set_ylabel('|Convergence Error| (eV)', fontweight='bold')
    ax2.set_title('Absolute Convergence Error Comparison', fontweight='bold', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(basis_sets, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale to show small differences
    
    # Analysis text
    if len(abs_h2o) > 2 and len(abs_reaction) > 2:
        h2o_final = abs_h2o[-2]  # Second to last (avoid potential artifacts)
        reaction_final = abs_reaction[-2]
        
        if reaction_final < h2o_final:
            analysis_text = "✓ Reaction energy converges\nfaster due to error cancellation"
            text_color = 'green'
        else:
            analysis_text = "⚠ Unexpected convergence pattern"
            text_color = 'orange'
        
        ax2.text(0.02, 0.98, analysis_text, transform=ax2.transAxes, 
                fontsize=11, verticalalignment='top', color=text_color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        save_path = plots_dir / 'convergence_analysis.png'
        plt.savefig(save_path)
        print(f"✅ Saved: {save_path}")
    
    plt.show()
    return fig

def plot_final_summary(all_results, save=True):
    """
    Create comprehensive summary plot with key findings
    """
    
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Method accuracy summary (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'T7' in all_results and all_results['T7']:
        methods = []
        errors = []
        for method, result in all_results['T7'].items():
            if result is not None:
                methods.append(method)
                errors.append(abs(result['reaction_energy_ev'] - 4.92))
        
        if methods:
            colors = ['green' if e < 0.1 else 'orange' if e < 0.3 else 'red' for e in errors]
            bars = ax1.bar(methods, errors, color=colors, alpha=0.7)
            ax1.set_ylabel('|Error| (eV)')
            ax1.set_title('Method Accuracy', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
    
    # Plot 2: Basis convergence summary (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'T5' in all_results and all_results['T5']:
        basis_names = []
        energies = []
        for basis, result in all_results['T5'].items():
            if result is not None:
                basis_names.append(basis)
                energies.append(result['reaction_energy_ev'])
        
        if energies:
            ax2.plot(range(len(basis_names)), energies, 'o-', linewidth=2)
            ax2.axhline(y=4.92, color='red', linestyle='--', alpha=0.7)
            ax2.set_ylabel('Energy (eV)')
            ax2.set_title('Basis Convergence', fontweight='bold')
            ax2.set_xticks(range(len(basis_names)))
            ax2.set_xticklabels([b.replace('cc-pV', '') for b in basis_names], rotation=45)
            ax2.grid(True, alpha=0.3)
    
    # Plot 3: Thermodynamic breakdown (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'T8' in all_results and all_results['T8']:
        thermo = all_results['T8']
        if 'reaction_delta_g_ev' in thermo:
            electronic = thermo['electronic_results']['reaction_energy_ev']
            thermodynamic = thermo['reaction_delta_g_ev']
            
            components = ['Electronic', 'Thermodynamic', 'Experimental']
            values = [electronic, thermodynamic, 4.92]
            colors = ['blue', 'purple', 'red']
            
            bars = ax3.bar(components, values, color=colors, alpha=0.7)
            ax3.set_ylabel('Energy (eV)')
            ax3.set_title('Final Results', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance statistics (middle left, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    if 'T9' in all_results and all_results['T9'] and 'analysis_data' in all_results['T9']:
        analysis_data = all_results['T9']['analysis_data']
        methods = [d['method'] for d in analysis_data]
        errors = [d['error'] for d in analysis_data]
        abs_errors = [abs(d['error']) for d in analysis_data]
        
        # Sort by accuracy
        sorted_data = sorted(zip(methods, errors, abs_errors), key=lambda x: x[2])
        methods, errors, abs_errors = zip(*sorted_data)
        
        colors = ['green' if abs(e) < 0.043 else 'blue' if abs(e) < 0.1 
                 else 'orange' if abs(e) < 0.3 else 'red' for e in errors]
        
        bars = ax4.barh(methods, errors, color=colors, alpha=0.7)
        ax4.axvline(x=0, color='black', linewidth=1)
        ax4.axvline(x=0.043, color='green', linestyle=':', alpha=0.7, label='Chemical accuracy')
        ax4.axvline(x=-0.043, color='green', linestyle=':', alpha=0.7)
        ax4.set_xlabel('Error vs Experiment (eV)')
        ax4.set_title('Method Performance Ranking', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add error values
        for bar, error in zip(bars, errors):
            width = bar.get_width()
            ax4.text(width + 0.01 if width >= 0 else width - 0.01, bar.get_y() + bar.get_height()/2,
                    f'{error:+.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
    
    # Plot 5: Key findings text (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    findings_text = "KEY FINDINGS:\n\n"
    
    if 'T8' in all_results and all_results['T8']:
        final_error = all_results['T8'].get('error_vs_experimental', 0)
        performance = all_results['T8'].get('performance_rating', 'Unknown')
        findings_text += f"• Final ΔG error: {final_error:+.3f} eV\n"
        findings_text += f"• Performance: {performance}\n\n"
    
    if 'T9' in all_results and all_results['T9']:
        chem_acc = all_results['T9'].get('chemical_accurate_count', 0)
        total_methods = len(all_results['T9'].get('analysis_data', []))
        if total_methods > 0:
            findings_text += f"• Chemical accuracy: {chem_acc}/{total_methods} methods\n"
    
    findings_text += f"• Reaction: 2H₂O → 2H₂ + O₂\n"
    findings_text += f"• Temperature: 298.15 K\n"
    findings_text += f"• Reference: {4.92:.2f} eV"
    
    ax5.text(0.05, 0.95, findings_text, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Plot 6: Molecular geometries comparison (bottom, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    if 'T2' in all_results and all_results['T2']:
        molecules = []
        exp_values = []
        calc_values = []
        errors = []
        
        geom_data = all_results['T2']
        
        # H2 bond length
        if 'H2' in geom_data and 'comparisons' in geom_data['H2']:
            if 'bond_length' in geom_data['H2']['comparisons']:
                molecules.append('H₂ bond')
                exp_values.append(geom_data['H2']['comparisons']['bond_length']['experimental'])
                calc_values.append(geom_data['H2']['comparisons']['bond_length']['calculated'])
                errors.append(geom_data['H2']['comparisons']['bond_length']['percent_error'])
        
        # O2 bond length
        if 'O2' in geom_data and 'comparisons' in geom_data['O2']:
            if 'bond_length' in geom_data['O2']['comparisons']:
                molecules.append('O₂ bond')
                exp_values.append(geom_data['O2']['comparisons']['bond_length']['experimental'])
                calc_values.append(geom_data['O2']['comparisons']['bond_length']['calculated'])
                errors.append(geom_data['O2']['comparisons']['bond_length']['percent_error'])
        
        # H2O properties
        if 'H2O' in geom_data and 'comparisons' in geom_data['H2O']:
            if 'oh_bond_length' in geom_data['H2O']['comparisons']:
                molecules.append('H₂O O-H')
                exp_values.append(geom_data['H2O']['comparisons']['oh_bond_length']['experimental'])
                calc_values.append(geom_data['H2O']['comparisons']['oh_bond_length']['calculated'])
                errors.append(geom_data['H2O']['comparisons']['oh_bond_length']['percent_error'])
            
            if 'hoh_angle' in geom_data['H2O']['comparisons']:
                molecules.append('H₂O angle')
                exp_values.append(geom_data['H2O']['comparisons']['hoh_angle']['experimental'])
                calc_values.append(geom_data['H2O']['comparisons']['hoh_angle']['calculated'])
                errors.append(geom_data['H2O']['comparisons']['hoh_angle']['percent_error'])
        
        if molecules:
            x = np.arange(len(molecules))
            width = 0.35
            
            bars1 = ax6.bar(x - width/2, exp_values, width, label='Experimental', 
                           color='red', alpha=0.7)
            bars2 = ax6.bar(x + width/2, calc_values, width, label='Calculated', 
                           color='blue', alpha=0.7)
            
            ax6.set_ylabel('Value')
            ax6.set_title('Geometry Optimization: Experimental vs Calculated', fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(molecules)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            # Add error percentages
            for i, error in enumerate(errors):
                ax6.text(i, max(exp_values[i], calc_values[i]) * 1.05,
                        f'{error:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('DFT Water Splitting Analysis: Comprehensive Summary', 
                fontsize=16, fontweight='bold', y=0.98)
    
    if save:
        save_path = plots_dir / 'comprehensive_summary.png'
        plt.savefig(save_path)
        print(f"✅ Saved: {save_path}")
    
    plt.show()
    return fig

# Convenience function to generate all plots
def generate_all_plots(all_results):
    """Generate all scientific plots for the DFT analysis report"""
    
    print("\n" + "="*50)
    print("GENERATING SCIENTIFIC PLOTS FOR REPORT")
    print("="*50)
    
    figures = []
    
    # Plot 1: Basis set convergence
    if 'T5' in all_results and all_results['T5']:
        print("\n📊 Generating basis set convergence plots...")
        fig1 = plot_basis_set_convergence(all_results['T5'])
        if fig1:
            figures.append(fig1)
    
    # Plot 2: Method performance analysis
    if 'T7' in all_results and all_results['T7']:
        print("\n📊 Generating method performance analysis...")
        fig2 = plot_method_performance(all_results['T7'])
        if fig2:
            figures.append(fig2)
    
    # Plot 3: Thermodynamic contributions
    if 'T8' in all_results and all_results['T8']:
        print("\n📊 Generating thermodynamic analysis...")
        fig3 = plot_thermodynamic_contributions(all_results['T8'])
        if fig3:
            figures.append(fig3)
    
    # Plot 4: Convergence analysis
    if 'T6' in all_results and all_results['T6']:
        print("\n📊 Generating convergence analysis...")
        fig4 = plot_convergence_analysis(all_results['T6'])
        if fig4:
            figures.append(fig4)
    
    # Plot 5: Comprehensive summary
    print("\n📊 Generating comprehensive summary...")
    fig5 = plot_final_summary(all_results)
    if fig5:
        figures.append(fig5)
    
    print(f"\n✅ Generated {len(figures)} scientific plots")
    print(f"📁 All plots saved to: {plots_dir}")
    print("\nPlots suitable for research report:")
    print("• basis_set_convergence.png - Shows convergence behavior")
    print("• method_performance_analysis.png - Method comparison & Jacob's ladder")
    print("• thermodynamic_contributions.png - Thermodynamic breakdown")
    print("• convergence_analysis.png - Error cancellation analysis")
    print("• comprehensive_summary.png - Complete results overview")
    
    return figures

if __name__ == "__main__":
    print("DFT Analysis Plotting Module")
    print("Use generate_all_plots(results) to create all scientific plots")
