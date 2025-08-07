import pandas as pd
import numpy as np
from typing import List, Dict
import re

class NumericQueryEngine:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.numeric_columns = list(
            dataframe.select_dtypes(include=[np.number]).columns
        )

        # Extract units for each column
        self.units = self._extract_units()
        self.column_map = self._build_column_map()

    def _extract_units(self) -> Dict[str, str]:
        """Extract units from column names"""
        units = {}
        unit_pattern = re.compile(r"\(([^)]+)\)")

        for col in self.df.columns:
            match = unit_pattern.search(col)
            if match:
                base_name = col.split("(")[0].strip()
                units[base_name] = match.group(1)
                units[col] = match.group(1)

        return units

    def _build_column_map(self) -> Dict[str, str]:
        """Build mapping for numeric columns"""
        column_map = {}
        for col in self.numeric_columns:
            base_name = col.split("(")[0].strip().lower()
            column_map[base_name] = col
        return column_map

    def find_exact_value(
        self, column: str, value: float, tolerance: float = 1e-6
    ) -> pd.DataFrame:
        """Find rows with exact numeric value (with precision tolerance)"""
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric")

        return self.df[np.abs(self.df[column] - value) <= tolerance]

    def find_value_range(
        self, column: str, min_val: float, max_val: float, inclusive: bool = True
    ) -> pd.DataFrame:
        """Find values within numeric range"""
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric")

        if inclusive:
            return self.df[(self.df[column] >= min_val) & (self.df[column] <= max_val)]
        else:
            return self.df[(self.df[column] > min_val) & (self.df[column] < max_val)]

    def find_nearest_values(
        self, column: str, target: float, n: int = 5
    ) -> pd.DataFrame:
        """Find N nearest values to target"""
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric")

        distances = np.abs(self.df[column] - target)
        nearest_indices = distances.nsmallest(n).index
        return self.df.loc[nearest_indices].sort_values(column)

    def find_percentile_range(
        self, column: str, lower_percentile: float, upper_percentile: float
    ) -> pd.DataFrame:
        """Find values within percentile range"""
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric")

        lower_val = self.df[column].quantile(lower_percentile / 100)
        upper_val = self.df[column].quantile(upper_percentile / 100)

        return self.df[(self.df[column] >= lower_val) & (self.df[column] <= upper_val)]

    def find_outliers(
        self, column: str, method: str = "iqr", factor: float = 1.5
    ) -> pd.DataFrame:
        """Find outliers using IQR or standard deviation method"""
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric")

        if method == "iqr":
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return self.df[
                (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
            ]

        elif method == "std":
            mean = self.df[column].mean()
            std = self.df[column].std()
            return self.df[np.abs(self.df[column] - mean) > factor * std]

    def interpolate_between_points(
        self, x_column: str, y_column: str, x_target: float, method: str = "linear"
    ) -> float:
        """Interpolate y value for given x value"""
        if x_column not in self.numeric_columns or y_column not in self.numeric_columns:
            raise ValueError("Both columns must be numeric")

        # Sort by x_column for interpolation
        sorted_df = self.df.sort_values(x_column)

        if method == "linear":
            return np.interp(x_target, sorted_df[x_column], sorted_df[y_column])
        elif method == "cubic":
            from scipy.interpolate import interp1d

            f = interp1d(sorted_df[x_column], sorted_df[y_column], kind="cubic")
            return float(f(x_target))

    def find_crossing_points(self, column: str, threshold: float) -> pd.DataFrame:
        """Find points where values cross a threshold"""
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric")

        # Find sign changes
        diff = self.df[column] - threshold
        sign_changes = np.diff(np.sign(diff))
        crossing_indices = np.where(sign_changes != 0)[0]

        # Return rows around crossing points
        result_indices = []
        for idx in crossing_indices:
            result_indices.extend([idx, idx + 1])

        return self.df.iloc[result_indices].drop_duplicates()

    def calculate_numeric_statistics(
        self, columns: List[str] = None
    ) -> Dict[str, Dict]:
        """Calculate comprehensive statistics for numeric columns"""
        if columns is None:
            columns = self.numeric_columns

        stats = {}
        for col in columns:
            if col in self.numeric_columns:
                stats[col] = {
                    "count": self.df[col].count(),
                    "mean": self.df[col].mean(),
                    "median": self.df[col].median(),
                    "std": self.df[col].std(),
                    "min": self.df[col].min(),
                    "max": self.df[col].max(),
                    "range": self.df[col].max() - self.df[col].min(),
                    "q25": self.df[col].quantile(0.25),
                    "q75": self.df[col].quantile(0.75),
                    "iqr": self.df[col].quantile(0.75) - self.df[col].quantile(0.25),
                    "skewness": self.df[col].skew(),
                    "kurtosis": self.df[col].kurtosis(),
                    "unit": self.units.get(col, "unknown"),
                }

        return stats

    def convert_units(
        self, column: str, from_unit: str, to_unit: str, conversion_factor: float
    ) -> pd.Series:
        """Convert units for a numeric column"""
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric")

        return self.df[column] * conversion_factor

    def query_numeric_natural_language(self, query: str) -> pd.DataFrame:
        """Parse natural language queries for numeric operations"""
        query = query.lower().strip()

        # Extract numbers
        numbers = [float(x) for x in re.findall(r"\d+\.?\d*", query)]

        # Find column
        target_col = None
        for term, col in self.column_map.items():
            if term in query:
                target_col = col
                break

        if not target_col or not numbers:
            return pd.DataFrame()

        # Parse numeric query types
        if "exactly" in query or "precise" in query:
            return self.find_exact_value(target_col, numbers[0], tolerance=1e-10)
        elif "between" in query and len(numbers) >= 2:
            return self.find_value_range(target_col, min(numbers), max(numbers))
        elif "nearest" in query or "closest" in query:
            n = int(numbers[1]) if len(numbers) > 1 else 5
            return self.find_nearest_values(target_col, numbers[0], n)
        elif "percentile" in query and len(numbers) >= 2:
            return self.find_percentile_range(target_col, numbers[0], numbers[1])
        elif "outlier" in query:
            return self.find_outliers(target_col)
        else:
            return self.find_exact_value(target_col, numbers[0])
