"""
Data access utilities for the Pipeline data analysis agent.
This module provides dataset access functions to avoid circular imports.
"""

import os
import pandas as pd
from typing import Optional

# Global dataset variable
_DATASET = None

def get_dataset(remove_duplicates: bool):
    """Get the dataset for tools to use. Avoids serialization issues in web interface.

    Args:
        remove_duplicates: If True, removes exact duplicate rows. Default False for backward compatibility.
                          Recommended True for volume calculations, False for transaction counts.
    """
    global _DATASET
    if _DATASET is None:
        # Fallback: load dataset if not already loaded
        data_path = os.getenv("DATASET_PATH", "data/pipeline_data.parquet")
        try:
            _DATASET = pd.read_parquet(data_path)
            # Convert date columns
            if 'eff_gas_day' in _DATASET.columns:
                _DATASET['eff_gas_day'] = pd.to_datetime(_DATASET['eff_gas_day'])
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {data_path}: {e}")

    # Return deduplicated copy if requested (don't modify global dataset)
    if remove_duplicates:
        return _DATASET.drop_duplicates()
    return _DATASET

def set_dataset(dataset: pd.DataFrame):
    """Set the dataset globally. Used by agent setup."""
    global _DATASET
    _DATASET = dataset
