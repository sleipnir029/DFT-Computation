import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Import task modules
from task1_coordinates import Task1Coordinates
from task2_geometry import Task2Geometry
from task3_reaction_energy import Task3ReactionEnergy
from task4_scf_energies import Task4SCFEnergies
from task5_basis_convergence import Task5BasisConvergence
from task6_convergence_comparison import Task6ConvergenceComparison
from task7_method_ladder import Task7MethodLadder
from task8_thermodynamics import Task8Thermodynamics
from task9_accuracy_analysis import Task9AccuracyAnalysis

# Constants
HARTREE2EV = 27.211386
HARTREE2KJ = 2625.499638
KCAL2HARTREE = 0.00159362
AVOGADRO = 6.02214076e23
CALORIE2JOULE = 4.184

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WaterSplittingAnalysis:
    def __init__(self):
        self.results = {}
        self.constants = {
            'HARTREE2EV': HARTREE2EV,
            'HARTREE2KJ': HARTREE2KJ,
            'KCAL2HARTREE': KCAL2HARTREE,
            'AVOGADRO': AVOGADRO,
            'CALORIE2JOULE': CALORIE2JOULE
        }
        
        # Initialize task objects
        self.task1 = Task1Coordinates(self.constants)
        self.task2 = Task2Geometry(self.constants)
        self.task3 = Task3ReactionEnergy(self.constants)
        self.task4 = Task4SCFEnergies(self.constants)
        self.task5 = Task5BasisConvergence(self.constants)
        self.task6 = Task6ConvergenceComparison(self.constants)
        self.task7 = Task7MethodLadder(self.constants)
        self.task8 = Task8Thermodynamics(self.constants)
        self.task9 = Task9AccuracyAnalysis(self.constants)
    
    def run_individual_task(self, task_num):
        """Run a specific task"""
        if task_num == 1:
            return self.task1.run(self.results)
        elif task_num == 2:
            return self.task2.run(self.results)
        elif task_num == 3:
            return self.task3.run(self.results)
        elif task_num == 4:
            return self.task4.run(self.results)
        elif task_num == 5:
            return self.task5.run(self.results)
        elif task_num == 6:
            return self.task6.run(self.results)
        elif task_num == 7:
            return self.task7.run(self.results)
        elif task_num == 8:
            return self.task8.run(self.results)
        elif task_num == 9:
            return self.task9.run(self.results)
        else:
            print(f"Task {task_num} not found!")
            return None
    
    def run_all_tasks(self):
        """Run all tasks in sequence"""
        print("Starting complete DFT water splitting analysis...")
        
        for i in range(1, 10):
            print(f"\n{'='*50}")
            print(f"RUNNING TASK {i}")
            print('='*50)
            
            result = self.run_individual_task(i)
            if result is not None:
                self.results[f'task{i}'] = result
                print(f"Task {i} completed successfully!")
            else:
                print(f"Task {i} failed!")
                break
        
        # Save complete results
        with open('json/complete_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print("\n" + "="*50)
        print("ALL TASKS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return self.results
    
    def run_tasks_range(self, start, end):
        """Run tasks from start to end"""
        for i in range(start, end + 1):
            result = self.run_individual_task(i)
            if result is not None:
                self.results[f'task{i}'] = result
        return self.results

if __name__ == "__main__":
    analysis = WaterSplittingAnalysis()
    
    # Run all tasks
    results = analysis.run_all_tasks()
    
    # Or run individual tasks
    # results = analysis.run_individual_task(1)
    
    # Or run a range of tasks
    # results = analysis.run_tasks_range(1, 5)
