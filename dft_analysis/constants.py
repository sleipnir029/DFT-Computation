"""
Physical constants and molecular data for DFT water splitting analysis
CORRECTED VERSION with proper NIST data
Based on CODATA 2018 and NIST data
"""

# Energy conversion constants (CODATA 2018 - HIGH PRECISION)
HARTREE2EV = 27.211386245988
EV2HARTREE = 1.0 / HARTREE2EV
HARTREE2KCAL = 627.509608030593
KCAL2HARTREE = 1.0 / HARTREE2KCAL

# Universal constants
R_GAS = 8.314462618  # J mol^-1 K^-1
AVOGADRO = 6.02214076e23  # mol^-1
STANDARD_TEMP = 298.15  # K

# Experimental geometries for initial coordinates (Task T1)
EXPERIMENTAL_GEOMETRIES = {
    'H2': {
        'atoms': ['H', 'H'],
        'coordinates': [
            [0.0, 0.0, 0.0],
            [0.741, 0.0, 0.0]  # Experimental H-H bond length: 0.741 Å
        ],
        'charge': 0,
        'spin': 0,
        'reference': 'NIST Chemistry WebBook'
    },
    'O2': {
        'atoms': ['O', 'O'],
        'coordinates': [
            [0.0, 0.0, 0.0],
            [1.208, 0.0, 0.0]  # Experimental O-O bond length: 1.208 Å
        ],
        'charge': 0,
        'spin': 2,  # Triplet ground state
        'reference': 'NIST Chemistry WebBook'
    },
    'H2O': {
        'atoms': ['O', 'H', 'H'],
        'coordinates': [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.467],
            [0.0, -0.757, -0.467]  # Experimental geometry: r(OH) = 0.958 Å, angle = 104.5°
        ],
        'charge': 0,
        'spin': 0,
        'reference': 'NIST Chemistry WebBook'
    }
}

# Basis sets for convergence study (Task T5)
BASIS_SETS = ['STO-3G', '3-21G', '6-31G', 'cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ', 'cc-pV5Z']

# Methods for Jacob's ladder (Task T7)
DFT_METHODS = ['HF', 'MP2', 'CCSD']
DFT_FUNCTIONALS = ['PBE', 'PBE0']  # GGA -> Hybrid

# CORRECTED NIST thermodynamic data at 298.15 K (Task T8)
# Source: NIST Chemistry WebBook, CODATA values
NIST_THERMO_DATA = {
    'H2': {
        'entropy_gas': 130.68,        # J mol^-1 K^-1 (CODATA, CORRECTED)
        'zpe': 6.197,                 # kcal mol^-1 
        'thermal_enthalpy': 2.024     # kcal mol^-1
    },
    'O2': {
        'entropy_gas': 205.152,       # J mol^-1 K^-1 (CODATA, CORRECTED)
        'zpe': 0.988,                 # kcal mol^-1
        'thermal_enthalpy': 2.024     # kcal mol^-1
    },
    'H2O': {
        'entropy_liquid': 69.95,      # J mol^-1 K^-1 (liquid water, CORRECTED)
        'entropy_gas': 188.84,        # J mol^-1 K^-1 (gas phase)
        'zpe': 13.435,                # kcal mol^-1
        'thermal_enthalpy': 2.368     # kcal mol^-1
    }
}

# Literature reference values
EXPERIMENTAL_WATER_SPLITTING_DG = 4.92  # eV, from project description
EXPERIMENTAL_BOND_LENGTHS = {
    'H2': 0.741,     # Å
    'O2': 1.208,     # Å  
    'H2O_OH': 0.958  # Å
}
EXPERIMENTAL_BOND_ANGLES = {
    'H2O_HOH': 104.5  # degrees
}

# Convergence criteria
SCF_CONVERGENCE = 1e-8
GEOM_CONVERGENCE = 1e-6
