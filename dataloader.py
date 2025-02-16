import dask.dataframe as dd
from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None

    def load_data(self):
        """Load large CSV files using Dask for efficient memory usage"""
        try:
            logger.info(f"Loading data from {self.data_path}")
            files = list(self.data_path.glob("*.csv"))
            logger.info(f"Found {len(files)} CSV files: {[f.name for f in files]}")
            
            # Read first few lines of first file to get columns
            sample_df = pd.read_csv(files[0], nrows=5)
            dtypes = {col: 'float64' if 'id' in col.lower() else 'object' 
                     for col in sample_df.columns}
            
            self.df = dd.read_csv(
                self.data_path / "*.csv",
                assume_missing=True,
                dtype=dtypes
            )
            
            logger.info(f"Data loaded successfully. Columns found: {list(self.df.columns)}")
            # Skip row counting as it's computationally expensive
            logger.info("Dataset loaded and ready for queries")
            
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise 
