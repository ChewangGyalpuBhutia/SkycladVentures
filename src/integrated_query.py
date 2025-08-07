import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import re
from engines.row_queries import RowQueryEngine
from engines.numeric_queries import NumericQueryEngine
from engines.column_queries import ColumnFilterEngine


class TableProcessor:
    def __init__(self):
        self.unit_pattern = re.compile(r"\(([^)]+)\)")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV, Excel or JSON"""
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")

        self.units = {}
        for col in df.columns:
            match = self.unit_pattern.search(col)
            if match:
                self.units[col.split("(")[0].strip()] = match.group(1)

        return df


class IntegratedQueryEngine:
    def __init__(self, file_path: str = None, dataframe: pd.DataFrame = None):
        """Initialize with either file path or dataframe"""
        if file_path:
            self.processor = TableProcessor()
            self.df = self.processor.load_data(file_path)
        elif dataframe is not None:
            self.df = dataframe
            self.processor = TableProcessor()
        else:
            raise ValueError("Either file_path or dataframe must be provided")

        # Initialize all query engines
        self.row_engine = RowQueryEngine(self.df)
        self.numeric_engine = NumericQueryEngine(self.df)
        self.column_engine = ColumnFilterEngine(self.df)

        # Combined column mapping
        self.all_columns = list(self.df.columns)
        self.numeric_columns = list(self.df.select_dtypes(include=[np.number]).columns)

    def query(
        self, user_query: str, query_type: str = "auto"
    ) -> Union[pd.DataFrame, Dict, float]:
        """
        Main query interface that routes to appropriate engine

        Args:
            user_query: Natural language query
            query_type: 'auto', 'row', 'numeric', 'column', or 'stats'
        """
        user_query = user_query.strip()

        if query_type == "auto":
            query_type = self._detect_query_type(user_query)

        try:
            if query_type == "row":
                return self._handle_row_query(user_query)
            elif query_type == "numeric":
                return self._handle_numeric_query(user_query)
            elif query_type == "column":
                return self._handle_column_query(user_query)
            elif query_type == "stats":
                return self._handle_stats_query(user_query)
            elif query_type == "info":
                return self._handle_info_query(user_query)
            else:
                return self._handle_general_query(user_query)
        except Exception as e:
            return pd.DataFrame()  # Return empty dataframe on error

    def _detect_query_type(self, query: str) -> str:
        """Automatically detect the type of query"""
        query_lower = query.lower()

        # Stats queries
        if any(
            word in query_lower
            for word in [
                "statistics",
                "stats",
                "summary",
                "describe",
                "mean",
                "average",
                "std",
                "variance",
            ]
        ):
            return "stats"

        # Info queries
        if any(
            word in query_lower
            for word in ["info", "information", "columns", "dtypes", "types"]
        ):
            return "info"

        # Column filtering queries
        if any(
            word in query_lower
            for word in ["select", "only", "exclude", "without", "columns"]
        ):
            return "column"

        # Numeric operations
        if any(
            word in query_lower
            for word in ["interpolate", "nearest", "closest", "percentile", "outlier"]
        ):
            return "numeric"

        # Row filtering (default for most queries)
        return "row"

    def _handle_row_query(self, query: str) -> pd.DataFrame:
        """Handle row-based queries"""
        return self.row_engine.query_natural_language(query)

    def _handle_numeric_query(self, query: str) -> Union[pd.DataFrame, float]:
        """Handle numeric-specific queries"""
        # Check if it's an interpolation query
        if "interpolate" in query.lower():
            return self._handle_interpolation(query)
        else:
            return self.numeric_engine.query_numeric_natural_language(query)

    def _handle_column_query(self, query: str) -> pd.DataFrame:
        """Handle column filtering queries"""
        return self.column_engine.filter_natural_language(query)

    def _handle_stats_query(self, query: str) -> Dict:
        """Handle statistical queries"""
        query_lower = query.lower()

        # Extract column name if specified
        target_col = None
        for col in self.numeric_columns:
            if (
                col.lower() in query_lower
                or col.split("(")[0].strip().lower() in query_lower
            ):
                target_col = col
                break

        if target_col:
            return self.numeric_engine.calculate_numeric_statistics([target_col])
        else:
            return self.numeric_engine.calculate_numeric_statistics()

    def _handle_info_query(self, query: str) -> Dict:
        """Handle information queries about the dataset"""
        return self.column_engine.get_column_info()

    def _handle_interpolation(self, query: str) -> float:
        """Handle interpolation queries"""
        # Extract numbers and column names
        numbers = [float(x) for x in re.findall(r"\d+\.?\d*", query)]

        # Simple interpolation pattern matching
        # This is a basic implementation - can be enhanced
        if len(numbers) >= 1:
            # Default to first numeric column for interpolation
            if len(self.numeric_columns) >= 2:
                return self.numeric_engine.interpolate_between_points(
                    self.numeric_columns[0], self.numeric_columns[1], numbers[0]
                )

        return 0.0

    def _handle_general_query(self, query: str) -> pd.DataFrame:
        """Handle general queries by trying different engines"""
        # Try row query first
        result = self.row_engine.query_natural_language(query)
        if not result.empty:
            return result

        # Try numeric query
        result = self.numeric_engine.query_numeric_natural_language(query)
        if isinstance(result, pd.DataFrame) and not result.empty:
            return result

        # Try column query
        result = self.column_engine.filter_natural_language(query)
        if not result.empty:
            return result

        return pd.DataFrame()

    def get_data_overview(self) -> Dict:
        """Get comprehensive overview of the dataset"""
        return {
            "shape": self.df.shape,
            "columns": self.all_columns,
            "numeric_columns": self.numeric_columns,
            "column_info": self.column_engine.get_column_info(),
            "summary_stats": self.numeric_engine.calculate_numeric_statistics(),
            "sample_data": self.df.head().to_dict(),
        }

    def suggest_queries(self) -> List[str]:
        """Suggest example queries based on the data"""
        suggestions = [
            f"Show me rows where {self.numeric_columns[0]} is greater than {self.df[self.numeric_columns[0]].mean():.2f}",
            f"Find the 5 highest values of {self.numeric_columns[0]}",
            f"What are the statistics for {self.numeric_columns[0]}?",
            "Show me only numeric columns",
            f"Find rows between {self.df[self.numeric_columns[0]].quantile(0.25):.2f} and {self.df[self.numeric_columns[0]].quantile(0.75):.2f} for {self.numeric_columns[0]}",
            "Get dataset information",
            f"Find outliers in {self.numeric_columns[0]}",
        ]

        return suggestions[:5]  # Return first 5 suggestions


# Convenience function for quick querying
def create_query_engine(file_path: str) -> IntegratedQueryEngine:
    """Create query engine from file"""
    return IntegratedQueryEngine(file_path=file_path)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the query engine
    engine = IntegratedQueryEngine(file_path="dataset/dataset.csv")

    # Example queries
    example_queries = [
        "Show me rows where draught is greater than 3",
        "Find the 5 highest displacement values",
        "What are the statistics for draught?",
        "Show me only numeric columns",
        "Get dataset information",
        "Find rows between 2 and 4 for draught",
        "Show closest values to 3.5 for draught",
    ]

    print("Dataset Overview:")
    print("=" * 50)
    overview = engine.get_data_overview()
    print(f"Shape: {overview['shape']}")
    print(f"Columns: {len(overview['columns'])}")
    print(f"Numeric Columns: {len(overview['numeric_columns'])}")

    print("\nExample Queries:")
    print("=" * 50)
    for query in example_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        result = engine.query(query)

        if isinstance(result, pd.DataFrame):
            if not result.empty:
                print(f"Found {len(result)} rows")
                print(result.head())
            else:
                print("No results found")
        elif isinstance(result, dict):
            print("Statistics/Info:")
            for key, value in list(result.items())[:3]:  # Show first 3 items
                print(f"{key}: {value}")
        else:
            print(f"Result: {result}")
