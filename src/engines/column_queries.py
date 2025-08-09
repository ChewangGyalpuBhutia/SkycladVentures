import pandas as pd
import numpy as np
from typing import List, Dict, Union
import re

class ColumnFilterEngine:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.all_columns = list(dataframe.columns)
        self.numeric_columns = list(dataframe.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(dataframe.select_dtypes(exclude=[np.number]).columns)
    
    def get_column_info(self) -> Dict:
        """Get comprehensive column information"""
        info = {
            "total_columns": len(self.all_columns),
            "numeric_columns": {
                "count": len(self.numeric_columns),
                "names": self.numeric_columns
            },
            "categorical_columns": {
                "count": len(self.categorical_columns), 
                "names": self.categorical_columns
            },
            "column_details": {}
        }
        
        # Add detailed info for each column
        for col in self.all_columns:
            col_info = {
                "dtype": str(self.df[col].dtype),
                "null_count": int(self.df[col].isnull().sum()),
                "unique_count": int(self.df[col].nunique()),
                "sample_values": self.df[col].dropna().head(3).tolist()
            }
            
            if col in self.numeric_columns:
                col_info.update({
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                    "mean": float(self.df[col].mean()),
                    "std": float(self.df[col].std())
                })
            
            info["column_details"][col] = col_info
        
        return info
    
    def filter_columns_by_type(self, include_types: List[str] = None, exclude_types: List[str] = None) -> pd.DataFrame:
        """Filter dataframe to include/exclude specific column types"""
        if include_types:
            if 'numeric' in include_types:
                return self.df[self.numeric_columns]
            elif 'categorical' in include_types:
                return self.df[self.categorical_columns]
        
        if exclude_types:
            cols_to_exclude = []
            if 'numeric' in exclude_types:
                cols_to_exclude.extend(self.numeric_columns)
            if 'categorical' in exclude_types:
                cols_to_exclude.extend(self.categorical_columns)
            
            remaining_cols = [col for col in self.all_columns if col not in cols_to_exclude]
            return self.df[remaining_cols]
        
        return self.df
    
    def select_columns(self, column_names: List[str]) -> pd.DataFrame:
        """Select specific columns by name"""
        # Find matching columns (case insensitive partial matching)
        selected_cols = []
        
        for requested_col in column_names:
            # Exact match first
            if requested_col in self.all_columns:
                selected_cols.append(requested_col)
                continue
            
            # Partial match (case insensitive)
            matches = [col for col in self.all_columns 
                      if requested_col.lower() in col.lower()]
            selected_cols.extend(matches)
        
        # Remove duplicates while preserving order
        selected_cols = list(dict.fromkeys(selected_cols))
        
        if selected_cols:
            return self.df[selected_cols]
        else:
            return pd.DataFrame()
    
    def exclude_columns(self, column_names: List[str]) -> pd.DataFrame:
        """Exclude specific columns by name"""
        # Find columns to exclude (case insensitive partial matching)
        cols_to_exclude = []
        
        for exclude_col in column_names:
            # Exact match first
            if exclude_col in self.all_columns:
                cols_to_exclude.append(exclude_col)
                continue
            
            # Partial match (case insensitive)
            matches = [col for col in self.all_columns 
                      if exclude_col.lower() in col.lower()]
            cols_to_exclude.extend(matches)
        
        # Remove duplicates
        cols_to_exclude = list(set(cols_to_exclude))
        
        # Get remaining columns
        remaining_cols = [col for col in self.all_columns if col not in cols_to_exclude]
        
        if remaining_cols:
            return self.df[remaining_cols]
        else:
            return pd.DataFrame()
    
    def filter_natural_language(self, query: str) -> pd.DataFrame:
        """Process natural language queries for column operations"""
        query = query.lower().strip()
        
        try:
            # Show only specific columns
            if 'only' in query or 'select' in query:
                # Extract column names or types
                if 'numeric' in query:
                    return self.filter_columns_by_type(['numeric'])
                elif 'categorical' in query or 'text' in query:
                    return self.filter_columns_by_type(['categorical'])
                else:
                    # Extract column names from query
                    mentioned_cols = []
                    for col in self.all_columns:
                        if col.lower() in query or col.split('(')[0].strip().lower() in query:
                            mentioned_cols.append(col)
                    
                    if mentioned_cols:
                        return self.select_columns(mentioned_cols)
            
            # Exclude columns
            elif 'without' in query or 'exclude' in query:
                # Extract column names or types
                if 'numeric' in query:
                    return self.filter_columns_by_type(exclude_types=['numeric'])
                elif 'categorical' in query or 'text' in query:
                    return self.filter_columns_by_type(exclude_types=['categorical'])
                else:
                    # Extract column names from query
                    mentioned_cols = []
                    for col in self.all_columns:
                        if col.lower() in query or col.split('(')[0].strip().lower() in query:
                            mentioned_cols.append(col)
                    
                    if mentioned_cols:
                        return self.exclude_columns(mentioned_cols)
            
            # Show column information
            elif any(word in query for word in ['info', 'information', 'columns', 'dtypes', 'types']):
                # This will be handled by the info query type, return empty df
                return pd.DataFrame()
                
        except Exception as e:
            return pd.DataFrame()
        
        return self.df