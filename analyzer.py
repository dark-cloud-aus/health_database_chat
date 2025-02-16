from typing import Dict, List
import plotly.express as px
import pandas as pd

class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def analyze_trends(self, columns: List[str]) -> Dict:
        """Analyze trends in specified columns"""
        results = {}
        for col in columns:
            try:
                # Compute basic trends
                trend_data = self.df[col].value_counts().compute()
                results[col] = {
                    'distribution': trend_data.to_dict(),
                    'summary': self.df[col].describe().compute().to_dict()
                }
            except Exception as e:
                results[col] = f"Error analyzing {col}: {str(e)}"
        return results

    def generate_visualization(self, column: str, chart_type: str = 'histogram'):
        """Generate visualizations for a column"""
        if chart_type == 'histogram':
            fig = px.histogram(self.df[column].compute())
        elif chart_type == 'box':
            fig = px.box(self.df[column].compute())
        return fig 
