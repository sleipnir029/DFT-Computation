"""
Physical constants with high precision values
All values from CODATA-2018 recommendations
"""

# Energy conversion factors
HARTREE2EV = 27.211_386_245_988      # CODATA-18, exact
HARTREE2KJ = 2625.499_639_48         # kJ/mol per Hartree
KCAL2HARTREE = 0.001_593_601_0974    # Hartree per kcal/mol (8 sig figs)
CALORIE2JOULE = 4.184                # exact definition

# Universal constants
R_GAS_CONSTANT = 8.314_462_618       # J mol⁻¹ K⁻¹, CODATA-2018
BOLTZMANN = 1.380_649e-23            # J K⁻¹, CODATA-2018
AVOGADRO = 6.022_140_76e23          # mol⁻¹, exact
PLANCK = 6.626_070_15e-34           # J⋅s, exact

# Temperature and pressure
STANDARD_TEMP = 298.15               # K
STANDARD_PRESSURE = 101325.0         # Pa (1 atm)

# Unit conversion dictionary for easy access
CONSTANTS = {
    'HARTREE2EV': HARTREE2EV,
    'HARTREE2KJ': HARTREE2KJ,
    'KCAL2HARTREE': KCAL2HARTREE,
    'R_GAS_CONSTANT': R_GAS_CONSTANT,
    'STANDARD_TEMP': STANDARD_TEMP
}
