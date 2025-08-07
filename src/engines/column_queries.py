import pandas as pd
import numpy as np
from typing import List, Dict
import re

class ColumnFilterEngine:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.columns = list(dataframe.columns)
        self.numeric_columns = list(
            dataframe.select_dtypes(include=[np.number]).columns
        )
        self.categorical_columns = list(
            dataframe.select_dtypes(exclude=[np.number]).columns
        )

        # Build column mapping
        self.column_map = self._build_column_map()

    def _build_column_map(self) -> Dict[str, str]:
        """Build mapping from common terms to actual column names"""
        column_map = {}

        for col in self.columns:
            # Extract base name (remove units)
            base_name = col.split("(")[0].strip().lower()
            column_map[base_name] = col

            # Add common aliases
            if "draught" in base_name:
                column_map["draft"] = col
                column_map["depth"] = col
            elif "disp" in base_name and "t" in col:
                column_map["displacement"] = col
            elif "volt" in base_name:
                column_map["volume"] = col
            elif base_name == "cb":
                column_map["block coefficient"] = col
                column_map["block_coefficient"] = col

        return column_map

    def select_columns(self, column_names: List[str]) -> pd.DataFrame:
        """Select specific columns from dataframe"""
        valid_columns = [col for col in column_names if col in self.df.columns]
        if not valid_columns:
            raise ValueError("No valid columns found")
        return self.df[valid_columns]

    def filter_by_column_type(self, column_type: str) -> pd.DataFrame:
        """Filter to show only numeric or categorical columns"""
        if column_type.lower() == "numeric":
            return self.df[self.numeric_columns]
        elif column_type.lower() == "categorical":
            return self.df[self.categorical_columns]
        else:
            raise ValueError("column_type must be 'numeric' or 'categorical'")

    def exclude_columns(self, column_names: List[str]) -> pd.DataFrame:
        """Exclude specific columns from dataframe"""
        remaining_cols = [col for col in self.columns if col not in column_names]
        return self.df[remaining_cols]

    def filter_columns_by_value_range(
        self, min_value: float = None, max_value: float = None
    ) -> pd.DataFrame:
        """Filter to columns where ALL values are within specified range"""
        valid_columns = []

        for col in self.numeric_columns:
            col_min = self.df[col].min()
            col_max = self.df[col].max()

            include_col = True
            if min_value is not None and col_min < min_value:
                include_col = False
            if max_value is not None and col_max > max_value:
                include_col = False

            if include_col:
                valid_columns.append(col)

        return self.df[valid_columns] if valid_columns else pd.DataFrame()

    def filter_columns_by_variance(self, min_variance: float = 0.0) -> pd.DataFrame:
        """Filter columns with variance above threshold (removes constant columns)"""
        valid_columns = []

        for col in self.numeric_columns:
            if self.df[col].var() > min_variance:
                valid_columns.append(col)

        return self.df[valid_columns] if valid_columns else pd.DataFrame()

    def filter_columns_by_correlation(
        self, target_column: str, min_correlation: float = 0.5
    ) -> pd.DataFrame:
        """Filter columns with correlation above threshold with target column"""
        if target_column not in self.numeric_columns:
            raise ValueError(
                f"Target column '{target_column}' not found or not numeric"
            )

        correlations = self.df[self.numeric_columns].corr()[target_column].abs()
        valid_columns = correlations[correlations >= min_correlation].index.tolist()

        return self.df[valid_columns] if valid_columns else pd.DataFrame()

    def get_columns_summary(self) -> pd.DataFrame:
        """Get summary statistics for all numeric columns"""
        return self.df[self.numeric_columns].describe()

    def get_column_info(self) -> Dict[str, Dict]:
        """Get detailed information about each column"""
        info = {}

        for col in self.columns:
            col_info = {
                "dtype": str(self.df[col].dtype),
                "non_null_count": self.df[col].count(),
                "null_count": self.df[col].isnull().sum(),
                "unique_values": self.df[col].nunique(),
            }

            if col in self.numeric_columns:
                col_info.update(
                    {
                        "min": self.df[col].min(),
                        "max": self.df[col].max(),
                        "mean": self.df[col].mean(),
                        "std": self.df[col].std(),
                    }
                )

            info[col] = col_info

        return info

    def resolve_column_names(self, user_inputs: List[str]) -> List[str]:
        """Resolve user input column names to actual column names"""
        resolved = []

        for user_input in user_inputs:
            user_input = user_input.lower().strip()

            # Direct match
            if user_input in self.column_map:
                resolved.append(self.column_map[user_input])
            else:
                # Partial match
                found = False
                for key, col in self.column_map.items():
                    if user_input in key or key in user_input:
                        resolved.append(col)
                        found = True
                        break

                if not found:
                    # Try exact column name match
                    exact_matches = [
                        col for col in self.columns if user_input in col.lower()
                    ]
                    if exact_matches:
                        resolved.append(exact_matches[0])

        return resolved

    def filter_natural_language(self, query: str) -> pd.DataFrame:
        """Parse natural language query for column filtering"""
        query = query.lower().strip()

        # Extract column names mentioned
        mentioned_columns = []
        for term in self.column_map.keys():
            if term in query:
                mentioned_columns.append(self.column_map[term])

        # Query patterns
        if "only" in query or "just" in query or "select" in query:
            if mentioned_columns:
                return self.select_columns(mentioned_columns)

        elif "exclude" in query or "without" in query or "except" in query:
            if mentioned_columns:
                return self.exclude_columns(mentioned_columns)

        elif "numeric" in query or "numbers" in query:
            return self.filter_by_column_type("numeric")

        elif "text" in query or "categorical" in query:
            return self.filter_by_column_type("categorical")

        elif "unit" in query:
            # Extract unit from query
            unit_match = re.search(r"unit[s]?\s+(\w+)", query)
            if unit_match:
                return self.filter_columns_by_unit(unit_match.group(1))

        elif "correlated" in query or "correlation" in query:
            if mentioned_columns:
                return self.filter_columns_by_correlation(mentioned_columns[0])

        # Default: return mentioned columns if any
        if mentioned_columns:
            return self.select_columns(mentioned_columns)

        return pd.DataFrame()
