import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import re

class NumericQueryEngine:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.numeric_columns = list(dataframe.select_dtypes(include=[np.number]).columns)
    
    def calculate_numeric_statistics(self, columns: List[str] = None) -> Dict:
        """Calculate comprehensive statistics for numeric columns"""
        if columns is None:
            columns = self.numeric_columns
        
        if not columns:
            return {"error": "No numeric columns found"}
        
        stats = {}
        
        for col in columns:
            if col in self.df.columns:
                col_stats = self.df[col].describe()
                # Convert numpy types to Python native types
                stats[col] = {
                    "count": int(col_stats['count']),
                    "mean": float(col_stats['mean']),
                    "std": float(col_stats['std']),
                    "min": float(col_stats['min']),
                    "25%": float(col_stats['25%']),
                    "50%": float(col_stats['50%']),
                    "75%": float(col_stats['75%']),
                    "max": float(col_stats['max']),
                    "variance": float(self.df[col].var()),
                    "skewness": float(self.df[col].skew()),
                    "kurtosis": float(self.df[col].kurtosis())
                }
        
        return stats
    
    def find_outliers_iqr(self, column: str, multiplier: float = 1.5) -> pd.DataFrame:
        """Find outliers using IQR method"""
        if column not in self.numeric_columns:
            return pd.DataFrame()
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers
    
    def find_values_near(self, column: str, target_value: float, tolerance: float = None) -> pd.DataFrame:
        """Find values near a target value"""
        if column not in self.numeric_columns:
            return pd.DataFrame()
        
        if tolerance is None:
            tolerance = self.df[column].std() * 0.1
        
        mask = abs(self.df[column] - target_value) <= tolerance
        return self.df[mask].sort_values(by=column)
    
    def interpolate_between_points(self, x_column: str, y_column: str, x_value: float) -> float:
        """Interpolate y value for given x value"""
        if x_column not in self.numeric_columns or y_column not in self.numeric_columns:
            return 0.0
        
        # Sort by x column
        sorted_df = self.df.sort_values(by=x_column)
        
        # Use numpy interp for interpolation
        result = np.interp(x_value, sorted_df[x_column], sorted_df[y_column])
        return float(result)  # Convert to Python float
    
    def find_percentile_values(self, column: str, percentiles: List[float]) -> Dict[str, float]:
        """Find values at specific percentiles"""
        if column not in self.numeric_columns:
            return {}
        
        result = {}
        for p in percentiles:
            result[f"{p}th_percentile"] = float(self.df[column].quantile(p/100))
        
        return result
    
    def query_numeric_natural_language(self, query: str) -> Union[pd.DataFrame, Dict, float]:
        """Process natural language queries for numeric operations"""
        query = query.lower().strip()
        
        # Extract column names
        target_columns = []
        for col in self.numeric_columns:
            if col.lower() in query or col.split('(')[0].strip().lower() in query:
                target_columns.append(col)
        
        # If no specific column mentioned, use first numeric column
        if not target_columns and self.numeric_columns:
            target_columns = [self.numeric_columns[0]]
        
        # Extract numbers from query
        numbers = [float(x) for x in re.findall(r'\d+\.?\d*', query)]
        
        try:
            # Statistics queries
            if any(word in query for word in ['statistics', 'stats', 'summary', 'describe']):
                return self.calculate_numeric_statistics(target_columns)
            
            # Mean/average queries
            elif any(word in query for word in ['mean', 'average']):
                if target_columns:
                    result = {}
                    for col in target_columns:
                        result[f"{col}_mean"] = float(self.df[col].mean())
                    return result
                
            # Standard deviation queries
            elif 'std' in query or 'standard deviation' in query:
                if target_columns:
                    result = {}
                    for col in target_columns:
                        result[f"{col}_std"] = float(self.df[col].std())
                    return result
            
            # Outlier queries
            elif 'outlier' in query:
                if target_columns:
                    return self.find_outliers_iqr(target_columns[0])
            
            # Nearest/closest queries
            elif any(word in query for word in ['nearest', 'closest', 'near']) and numbers:
                if target_columns and numbers:
                    return self.find_values_near(target_columns[0], numbers[0])
            
            # Percentile queries
            elif 'percentile' in query and numbers:
                if target_columns:
                    return self.find_percentile_values(target_columns[0], numbers)
            
            # Interpolation queries
            elif 'interpolate' in query and len(target_columns) >= 2 and numbers:
                return self.interpolate_between_points(target_columns[0], target_columns[1], numbers[0])
                
        except Exception as e:
            return pd.DataFrame()
        
        return pd.DataFrame()