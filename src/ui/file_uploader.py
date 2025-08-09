# src/ui/file_uploader.py
import streamlit as st
import pandas as pd
import json
import io
from typing import Optional, Tuple
from pathlib import Path

class FileUploader:
    """Handle file upload and validation"""
    
    SUPPORTED_FORMATS = {
        'csv': ['.csv'],
        'json': ['.json'],
        'excel': ['.xlsx', '.xls']
    }
    
    MAX_FILE_SIZE_MB = 200  # Maximum file size in MB
    
    @staticmethod
    def render_file_uploader() -> Optional[pd.DataFrame]:
        """Render file upload interface and return loaded dataframe"""
        st.header("📁 Upload Your Dataset")
        
        # File upload widget
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'json', 'xlsx', 'xls'],
            help=f"Supported formats: CSV, JSON, Excel. Max size: {FileUploader.MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file is not None:
            return FileUploader._process_uploaded_file(uploaded_file)
        
        return None
    
    @staticmethod
    def _process_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
        """Process the uploaded file and return dataframe"""
        try:
            # Check file size
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > FileUploader.MAX_FILE_SIZE_MB:
                st.error(f"File too large! Maximum size is {FileUploader.MAX_FILE_SIZE_MB}MB. Your file is {file_size_mb:.1f}MB")
                return None
            
            # Show file info
            st.info(f"📄 **File:** {uploaded_file.name} ({file_size_mb:.1f}MB)")
            
            # Determine file type and load accordingly
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            with st.spinner("🔄 Loading file..."):
                if file_extension == '.csv':
                    df = FileUploader._load_csv(uploaded_file)
                elif file_extension == '.json':
                    df = FileUploader._load_json(uploaded_file)
                elif file_extension in ['.xlsx', '.xls']:
                    df = FileUploader._load_excel(uploaded_file)
                else:
                    st.error(f"Unsupported file format: {file_extension}")
                    return None
            
            if df is not None:
                FileUploader._display_file_preview(df, uploaded_file.name)
                return df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    @staticmethod
    def _load_csv(uploaded_file) -> Optional[pd.DataFrame]:
        """Load CSV file with encoding detection"""
        try:
            # Try UTF-8 first
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            return df
        except UnicodeDecodeError:
            try:
                # Try latin-1 encoding
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
                st.warning("⚠️ File loaded with latin-1 encoding")
                return df
            except Exception as e:
                st.error(f"Failed to load CSV: {str(e)}")
                return None
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return None
    
    @staticmethod
    def _load_json(uploaded_file) -> Optional[pd.DataFrame]:
        """Load JSON file"""
        try:
            # Read the file content
            content = uploaded_file.read()
            
            # Parse JSON
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            json_data = json.loads(content)
            
            # Convert to DataFrame
            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                # If it's a dict, try to convert to DataFrame
                df = pd.DataFrame([json_data])
            else:
                st.error("JSON format not supported. Expected list of objects or single object.")
                return None
            
            return df
            
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error reading JSON: {str(e)}")
            return None
    
    @staticmethod
    def _load_excel(uploaded_file) -> Optional[pd.DataFrame]:
        """Load Excel file"""
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(uploaded_file)
            
            # If multiple sheets, let user choose
            if len(excel_file.sheet_names) > 1:
                sheet_name = st.selectbox(
                    "Select sheet:",
                    excel_file.sheet_names,
                    key="excel_sheet_selector"
                )
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            else:
                df = pd.read_excel(uploaded_file)
            
            return df
            
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return None
    
    @staticmethod
    def _display_file_preview(df: pd.DataFrame, filename: str):
        """Display preview of loaded file"""
        st.success(f"✅ Successfully loaded {filename}")
        
        # File statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            st.metric("Numeric Columns", numeric_cols)
        with col4:
            missing_values = df.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        
        # Column information
        with st.expander("📋 Column Information"):
            col_info = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                
                col_info.append({
                    'Column': col,
                    'Type': col_type,
                    'Null Count': null_count,
                    'Unique Values': unique_count
                })
            
            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, use_container_width=True)
        
        # Data preview
        with st.expander("👀 Data Preview", expanded=True):
            preview_rows = st.slider("Preview rows", 5, min(50, len(df)), 10)
            st.dataframe(df.head(preview_rows), use_container_width=True)
        
        # Data quality checks
        FileUploader._display_data_quality_checks(df)
    
    @staticmethod
    def _display_data_quality_checks(df: pd.DataFrame):
        """Display data quality information"""
        with st.expander("🔍 Data Quality Checks"):
            
            # Missing values analysis
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.subheader("⚠️ Missing Values")
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': (missing_data.values / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found")
            
            # Duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                st.warning(f"⚠️ Found {duplicate_count} duplicate rows")
            else:
                st.success("✅ No duplicate rows found")
            
            # Column data types
            st.subheader("📊 Data Types Summary")
            dtype_summary = df.dtypes.value_counts().to_frame('Count')
            dtype_summary.index.name = 'Data Type'
            st.dataframe(dtype_summary, use_container_width=True)