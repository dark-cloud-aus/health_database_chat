import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, df):
        self.df = df
        
    def get_summary_stats(self) -> Dict:
        """Get basic summary statistics of the dataset"""
        try:
            # Get all numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            return {
                'columns': list(self.df.columns),
                'column_types': {col: str(self.df[col].dtype) for col in self.df.columns},
                'files_loaded': True
            }
        except Exception as e:
            logger.error(f"Error in get_summary_stats: {str(e)}")
            return {
                'columns': list(self.df.columns),
                'error': str(e)
            }
    
    def get_column_info(self, column: str) -> Dict:
        """Get detailed information about a specific column"""
        if column not in self.df.columns:
            return {'error': f'Column {column} not found in dataset'}
            
        return {
            'unique_values': self.df[column].nunique().compute(),
            'value_counts': self.df[column].value_counts().compute(),
            'data_type': str(self.df[column].dtype)
        } 
