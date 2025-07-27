"""
Task T1: Define experimental coordinates for H2, O2, and H2O
"""

import pandas as pd
from config.constants import EXPERIMENTAL_GEOMETRIES
from utils.data_manager import data_manager

def get_experimental_coordinates():
    """
    Returns experimental geometries for all three molecules
    
    Returns:
        dict: Dictionary containing experimental coordinates and references
    """
    
    print("=== Task T1: Experimental Coordinates ===")
    
    coordinates = {}
    csv_data = []
    
    for molecule, data in EXPERIMENTAL_GEOMETRIES.items():
        print(f"\n{molecule}:")
        print(f"  Reference: {data['reference']}")
        print(f"  Charge: {data['charge']}, Spin: {data['spin']}")
        print(f"  Atoms and coordinates (Å):")
        
        for i, (atom, coord) in enumerate(zip(data['atoms'], data['coordinates'])):
            print(f"    {atom}: {coord[0]:8.3f} {coord[1]:8.3f} {coord[2]:8.3f}")
            
            # Add to CSV data
            csv_data.append({
                'Molecule': molecule,
                'Atom_Index': i+1,
                'Atom_Type': atom,
                'X_Angstrom': coord[0],
                'Y_Angstrom': coord[1],
                'Z_Angstrom': coord[2],
                'Charge': data['charge'],
                'Spin': data['spin'],
                'Reference': data['reference']
            })
        
        coordinates[molecule] = {
            'atoms': data['atoms'],
            'coordinates': data['coordinates'],
            'charge': data['charge'],
            'spin': data['spin']
        }
    
    # Create CSV table
    csv_table = pd.DataFrame(csv_data)
    
    # Save data
    data_manager.save_task_data(
        'task1_coordinates',
        coordinates,
        {'coordinates': csv_table}
    )
    
    return coordinates

if __name__ == "__main__":
    coords = get_experimental_coordinates()
