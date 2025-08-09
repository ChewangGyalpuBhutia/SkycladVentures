import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import re

class RowQueryEngine:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.columns = list(dataframe.columns)

        # Create column mapping for natural language
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

    def get_exact_row(
        self, column: str, value: float, tolerance: float = 0.001
    ) -> pd.DataFrame:
        """Get row(s) where column equals value (with tolerance for floats)"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        return self.df[abs(self.df[column] - value) <= tolerance]

    def get_rows_in_range(
        self, column: str, min_val: float, max_val: float, inclusive: bool = True
    ) -> pd.DataFrame:
        """Get rows where column value is within specified range"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        if inclusive:
            return self.df[(self.df[column] >= min_val) & (self.df[column] <= max_val)]
        else:
            return self.df[(self.df[column] > min_val) & (self.df[column] < max_val)]

    def get_rows_by_condition(
        self, column: str, operator: str, value: float
    ) -> pd.DataFrame:
        """Get rows based on condition"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        operators = {
            ">": lambda x, v: x > v,
            "<": lambda x, v: x < v,
            ">=": lambda x, v: x >= v,
            "<=": lambda x, v: x <= v,
            "==": lambda x, v: x == v,
            "!=": lambda x, v: x != v,
        }

        if operator not in operators:
            raise ValueError(f"Unsupported operator: {operator}")

        return self.df[operators[operator](self.df[column], value)]

    def get_closest_row(self, column: str, target_value: float) -> pd.DataFrame:
        """Get the row with the closest value to target"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        idx = (self.df[column] - target_value).abs().idxmin()
        return self.df.loc[[idx]]

    def get_top_n_rows(
        self, column: str, n: int = 5, ascending: bool = False
    ) -> pd.DataFrame:
        """Get top N rows sorted by column"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        return (
            self.df.nlargest(n, column)
            if not ascending
            else self.df.nsmallest(n, column)
        )

    def get_rows_by_multiple_conditions(self, conditions: List[Dict]) -> pd.DataFrame:
        """Get rows matching multiple conditions
        conditions: [{'column': 'DRAUGHT(m)', 'operator': '>', 'value': 3.0}, ...]
        """
        mask = pd.Series([True] * len(self.df), index=self.df.index)

        for condition in conditions:
            col = condition["column"]
            op = condition["operator"]
            val = condition["value"]

            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found")

            if op == ">":
                mask &= self.df[col] > val
            elif op == "<":
                mask &= self.df[col] < val
            elif op == ">=":
                mask &= self.df[col] >= val
            elif op == "<=":
                mask &= self.df[col] <= val
            elif op == "==":
                mask &= self.df[col] == val
            elif op == "!=":
                mask &= self.df[col] != val

        return self.df[mask]

    def interpolate_value(
        self, target_column: str, ref_column: str, ref_value: float
    ) -> float:
        """Interpolate target_column value for a given ref_column value"""
        if ref_column not in self.df.columns or target_column not in self.df.columns:
            raise ValueError("Column not found")

        sorted_df = self.df.sort_values(ref_column)
        return np.interp(ref_value, sorted_df[ref_column], sorted_df[target_column])

    def resolve_column_name(self, user_input: str) -> Optional[str]:
        """Resolve user input to actual column name"""
        user_input = user_input.lower().strip()

        # Direct match
        if user_input in self.column_map:
            return self.column_map[user_input]

        # Partial match
        for key, col in self.column_map.items():
            if user_input in key or key in user_input:
                return col

        return None

    def query_natural_language(self, query: str) -> pd.DataFrame:
        """Parse natural language query and return results"""
        query = query.lower().strip()

        # Extract numbers
        numbers = [float(x) for x in re.findall(r"\d+\.?\d*", query)]

        # Find column mentioned in query
        target_col = None
        for term in self.column_map.keys():
            if term in query:
                target_col = self.column_map[term]
                break

        if not target_col:
            return pd.DataFrame()

        # Parse query type and execute
        if not numbers:
            return pd.DataFrame()

        if "between" in query and len(numbers) >= 2:
            return self.get_rows_in_range(target_col, min(numbers), max(numbers))
        elif any(phrase in query for phrase in ["greater than", "more than", "above"]):
            return self.get_rows_by_condition(target_col, ">", numbers[0])
        elif any(phrase in query for phrase in ["less than", "below", "under"]):
            return self.get_rows_by_condition(target_col, "<", numbers[0])
        elif any(phrase in query for phrase in ["at least", "minimum"]):
            return self.get_rows_by_condition(target_col, ">=", numbers[0])
        elif any(phrase in query for phrase in ["at most", "maximum"]):
            return self.get_rows_by_condition(target_col, "<=", numbers[0])
        elif any(phrase in query for phrase in ["closest", "nearest"]):
            return self.get_closest_row(target_col, numbers[0])
        elif any(phrase in query for phrase in ["lowest", "smallest", "bottom"]):
            n = int(numbers[0]) if numbers else 5
            return self.get_top_n_rows(target_col, n, ascending=True)
        elif any(phrase in query for phrase in ["highest", "largest", "biggest", "top"]):
            n = int(numbers[0]) if numbers else 5
            return self.get_top_n_rows(target_col, n, ascending=False)
        else:
            # Default to exact match
            return self.get_exact_row(target_col, numbers[0])
