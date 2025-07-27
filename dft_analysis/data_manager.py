"""
Data management utilities for DFT analysis
Handles JSON and CSV saving with organized structure
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Union, Optional

class DFTDataManager:
    """
    Comprehensive data management for DFT analysis results
    """
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.json_dir = self.base_dir / "json"
        self.csv_dir = self.base_dir / "csv"
        self.plots_dir = self.base_dir / "plots"
        
        # Create directories
        for directory in [self.json_dir, self.csv_dir, self.plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_task_data(self, task_name: str, data: Dict[str, Any], 
                      csv_tables: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Save task data in both JSON and CSV formats
        
        Args:
            task_name: Name of the task (e.g., 'task1_coordinates')
            data: Complete data dictionary to save as JSON
            csv_tables: Optional dictionary of DataFrames to save as CSV
        """
        
        # Add metadata
        data_with_metadata = {
            'task': task_name,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Save JSON
        json_file = self.json_dir / f"{task_name}.json"
        with open(json_file, 'w') as f:
            json.dump(data_with_metadata, f, indent=2, default=self._numpy_converter)
        
        print(f"✅ Saved JSON: {json_file}")
        
        # Save CSV tables if provided
        if csv_tables:
            for table_name, df in csv_tables.items():
                csv_file = self.csv_dir / f"{task_name}_{table_name}.csv"
                df.to_csv(csv_file, index=False)
                print(f"✅ Saved CSV: {csv_file}")
    
    def _numpy_converter(self, obj):
        """Convert numpy objects to JSON serializable formats"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return str(obj)
    
    def create_summary_table(self, all_results: Dict[str, Any]) -> pd.DataFrame:
        """Create comprehensive summary table"""
        
        summary_data = []
        
        # Extract key results from each task
        for task_name, task_data in all_results.items():
            if task_data is None:
                continue
                
            if task_name == 'T7':  # Method comparison
                for method, result in task_data.items():
                    if result is not None:
                        summary_data.append({
                            'Task': task_name,
                            'Method': method,
                            'Reaction_Energy_eV': result['reaction_energy_ev'],
                            'Error_vs_Exp_eV': result['error_vs_exp'],
                            'Type': 'Electronic'
                        })
            
            elif task_name == 'T8':  # Thermodynamics
                if 'reaction_delta_g_ev' in task_data:
                    summary_data.append({
                        'Task': task_name,
                        'Method': 'MP2+Thermo',
                        'Reaction_Energy_eV': task_data['reaction_delta_g_ev'],
                        'Error_vs_Exp_eV': task_data['error_vs_experimental'],
                        'Type': 'Thermodynamic'
                    })
        
        return pd.DataFrame(summary_data)
    
    def save_comprehensive_analysis(self, all_results: Dict[str, Any]):
        """Save complete analysis with summary"""
        
        # Save complete results as JSON
        complete_file = self.base_dir / "complete_dft_analysis.json"
        complete_data = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'description': 'Complete DFT Water Splitting Analysis'
            },
            'results': all_results
        }
        
        with open(complete_file, 'w') as f:
            json.dump(complete_data, f, indent=2, default=self._numpy_converter)
        
        print(f"✅ Saved complete analysis: {complete_file}")
        
        # Create and save summary table
        summary_df = self.create_summary_table(all_results)
        summary_file = self.base_dir / "dft_analysis_comprehensive_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"✅ Saved summary table: {summary_file}")
        
        return summary_df

# Global instance
data_manager = DFTDataManager()
