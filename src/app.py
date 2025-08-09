import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Data Query & Analysis Platform",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import your modules after streamlit config
try:
    from integrated_query import IntegratedQueryEngine
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #0d47a1;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        
    }
    .ai-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        color: #0d47a1;
    }
    .follow-up-indicator {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        font-size: 0.9rem;
        color: #0d47a1;
    }
    .conversation-stats {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #6c757d;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px dashed #dee2e6;
        margin: 1rem 0;
    }
    .welcome-message {
        background-color: #e8f5e8;
        padding: 2rem;
        border-radius: 1rem;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = {}

def load_csv_file(uploaded_file):
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

def load_json_file(uploaded_file):
    """Load JSON file"""
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        json_data = json.loads(content)
        
        if isinstance(json_data, list):
            df = pd.DataFrame(json_data)
        elif isinstance(json_data, dict):
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

def load_excel_file(uploaded_file):
    """Load Excel file"""
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        
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

def save_uploaded_file(uploaded_file, df):
    """Save uploaded file to dataset directory"""
    # Create dataset directory if it doesn't exist
    dataset_dir = Path(__file__).parent.parent / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    # Save file with original name
    file_path = dataset_dir / uploaded_file.name
    
    # Also save as CSV for consistency
    csv_path = dataset_dir / f"{Path(uploaded_file.name).stem}_uploaded.csv"
    df.to_csv(csv_path, index=False)
    
    # Save original file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ File saved to: {file_path}")
    st.info(f"📄 Also saved as CSV: {csv_path}")
    
    return str(file_path)

def process_uploaded_file(uploaded_file):
    """Process uploaded file and return dataframe"""
    try:
        # Check file size (200MB limit)
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > 200:
            st.error(f"File too large! Maximum size is 200MB. Your file is {file_size_mb:.1f}MB")
            return None
        
        # Show file info
        st.info(f"📄 **File:** {uploaded_file.name} ({file_size_mb:.1f}MB)")
        
        # Load file based on extension
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        with st.spinner("🔄 Loading file..."):
            if file_extension == '.csv':
                df = load_csv_file(uploaded_file)
            elif file_extension == '.json':
                df = load_json_file(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                df = load_excel_file(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
        
        if df is not None:
            # Save file to dataset directory
            saved_path = save_uploaded_file(uploaded_file, df)
            st.session_state.uploaded_filename = uploaded_file.name
            
            return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def create_engine_from_dataframe(df):
    """Create query engine from dataframe"""
    try:
        engine = IntegratedQueryEngine(dataframe=df)
        return engine, None
    except Exception as e:
        return None, str(e)

def render_file_uploader():
    """Render file upload interface"""
    st.header("📁 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'json', 'xlsx', 'xls'],
        help="Supported formats: CSV, JSON, Excel"
    )
    
    if uploaded_file is not None:
        return process_uploaded_file(uploaded_file)
    
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🚢 Data Query & Analysis Platform</h1>', unsafe_allow_html=True)
    
    # Check if we have data loaded
    if st.session_state.query_engine is None:
        show_welcome_screen()
    else:
        show_main_interface()

def show_welcome_screen():
    uploaded_df = render_file_uploader()
    
    if uploaded_df is not None:
        # Create query engine from uploaded data
        with st.spinner("🔧 Initializing query engine..."):
            engine, error = create_engine_from_dataframe(uploaded_df)
            
            if engine:
                st.session_state.query_engine = engine
                st.success("✅ Query engine ready! You can now start querying your data.")
                
                if engine.rag_enabled:
                    st.success("🤖 RAG functionality enabled - You can chat with your data!")
                else:
                    st.warning("⚠️ RAG functionality disabled (check GEMINI_API_KEY in .env file)")
                
                # Auto-rerun to show main interface
                st.rerun()
            else:
                st.error(f"❌ Failed to initialize engine: {error}")

def show_main_interface():
    """Show the main interface when data is loaded"""
    engine = st.session_state.query_engine
    
    # Sidebar with data info and controls
    with st.sidebar:
        st.header("📊 Current Dataset")
        
        # Show uploaded filename
        if st.session_state.uploaded_filename:
            st.write(f"**File:** {st.session_state.uploaded_filename}")
        
        query_type = "auto"
        use_rag = st.checkbox(
            "Force RAG",
            value=True,
            disabled=not engine.rag_enabled,
            help="Force use of RAG for query processing"
        )
        
        # File management
        if st.button("📁 Upload New Dataset", type="secondary"):
            # Reset all session state
            st.session_state.query_engine = None
            st.session_state.current_data = None
            st.session_state.chat_history = []
            st.session_state.uploaded_filename = None
            st.session_state.conversation_context = {}
            st.rerun()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Query Interface", "💬 AI Chat", "📋 Data Overview"])
    
    with tab1:
        query_interface(engine, query_type, use_rag)
    
    with tab2:
        if engine.rag_enabled:
            enhanced_chat_interface(engine)
        else:
            st.warning("💬 AI Chat requires RAG functionality. Please check your GEMINI_API_KEY in .env file.")
            st.info("You can still use the Query Interface to explore your data!")
    
    with tab3:
        data_overview(engine)

def query_interface(engine, query_type, use_rag):
    """Query interface tab"""
    st.header("🔍 Natural Language Query Interface")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        suggestions = engine.suggest_queries()
        for i, suggestion in enumerate(suggestions, 1):
            if st.button(f"{i}. {suggestion}", key=f"suggestion_{i}"):
                st.session_state.query_input = suggestion
    
    # Query input
    query = st.text_area(
        "Enter your query:",
        value=st.session_state.get('query_input', ''),
        height=100,
        placeholder="e.g., Show me the top 10 rows with highest values, Find outliers in column X, What is the average of column Y?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        execute_query = st.button("🚀 Execute", type="primary")
    
    if execute_query and query:
        with st.spinner("Processing query..."):
            try:
                result = engine.query(query, query_type, use_rag)
                display_query_result(result, query)
            except Exception as e:
                st.error(f"Query error: {str(e)}")

def display_query_result(result, query):
    """Display query results"""
    st.subheader("📋 Results")
    
    if isinstance(result, pd.DataFrame):
        if result.empty:
            st.warning("No data found matching your query.")
        else:
            st.success(f"Found {len(result)} matching records")
            
            # Store current data for visualization
            st.session_state.current_data = result
            
            # Display options
            col1, col2, col3 = st.columns(3)
            with col1:
                show_raw = st.checkbox("Show raw data", value=True)
            with col2:
                max_rows = st.number_input("Max rows to display", 1, 1000, 100)
            with col3:
                download_csv = st.download_button(
                    "📥 Download CSV",
                    result.to_csv(index=False),
                    f"query_result.csv",
                    "text/csv"
                )
            
            if show_raw:
                st.dataframe(result.head(max_rows), use_container_width=True)
    
    elif isinstance(result, dict):
        if "response" in result:  # RAG response
            st.subheader("🤖 AI Response")
            
            # Show follow-up indicator if applicable
            if result.get("is_follow_up", False):
                st.markdown(
                    '<div class="follow-up-indicator">🔗 This appears to be a follow-up question based on conversation history</div>',
                    unsafe_allow_html=True
                )
            
            st.markdown(f'<div class="ai-message">{result["response"]}</div>', unsafe_allow_html=True)
            
            if "data" in result:
                st.subheader("📊 Related Data")
                df = pd.DataFrame(result["data"])
                st.dataframe(df, use_container_width=True)
        else:  # Stats/info response
            st.subheader("📈 Results")
            st.json(result)
    
    elif isinstance(result, (int, float)):
        st.metric("Result", f"{result:.4f}")
    
    else:
        st.write("**Result:**", str(result))

def enhanced_chat_interface(engine):
    """Enhanced AI Chat interface with conversation history"""
    st.header("💬 Chat with Your Data")
    
    # Conversation summary
    if engine.rag_enabled and engine.rag_engine:
        conv_summary = engine.rag_engine.get_conversation_summary()
        if conv_summary["conversation_active"]:
            st.markdown(
                f'<div class="conversation-stats">'
                f'💬 <strong>Active Conversation:</strong> {conv_summary["total_turns"]} exchanges<br>'
                f'📝 <strong>Recent topics:</strong> {", ".join(conv_summary["recent_topics"])}'
                f'</div>',
                unsafe_allow_html=True
            )
    
    # Chat history with enhanced display
    for i, (user_msg, ai_response) in enumerate(st.session_state.chat_history):
        # Add timestamp and exchange number
        timestamp = datetime.now().strftime("%H:%M")
        st.markdown(
            f'<div class="chat-message user-message">'
            f'<strong>You ({timestamp}):</strong> {user_msg}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="chat-message ai-message">'
            f'<strong>AI:</strong> {ai_response}'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # Chat input with follow-up suggestions
    with st.form("chat_form"):
        user_input = st.text_area("Ask a question about your data:", height=100)
        
        # Add follow-up suggestion buttons if there's conversation history
        if st.session_state.chat_history:
            st.write("**Quick follow-ups:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.form_submit_button("📊 Show more details"):
                    user_input = "Can you show more details about that?"
            with col2:
                if st.form_submit_button("🔍 Analyze further"):
                    user_input = "Can you analyze this further?"
            with col3:
                if st.form_submit_button("❓ Explain more"):
                    user_input = "Can you explain that in more detail?"
        
        submitted = st.form_submit_button("💬 Send", type="primary")
        
        if submitted and user_input:
            with st.spinner("AI is thinking..."):
                try:
                    response = engine.chat_with_data(user_input)
                    st.session_state.chat_history.append((user_input, response))
                    st.rerun()
                except Exception as e:
                    st.error(f"Chat error: {str(e)}")
    
    # Chat management buttons
    col1 = st.columns(1)[0]
    with col1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            if engine.rag_enabled and engine.rag_engine:
                engine.rag_engine.clear_conversation_history()
            st.rerun()

def data_overview(engine):
    """Data overview tab"""
    st.header("📋 Dataset Overview")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", engine.df.shape[0])
    with col2:
        st.metric("Total Columns", engine.df.shape[1])
    with col3:
        st.metric("Numeric Columns", len(engine.numeric_columns))
    with col4:
        st.metric("Memory Usage", f"{engine.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Sample data with pagination
    st.subheader("👀 Sample Data")
    
    # Pagination controls
    rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    total_rows = len(engine.df)
    total_pages = (total_rows - 1) // rows_per_page + 1
    
    # Initialize page number in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # Pagination buttons
    col2, col4 = st.columns([1, 1])
    
    with col2:
        if st.button("◀️ Previous", disabled=st.session_state.current_page == 1):
            st.session_state.current_page -= 1
            st.rerun()
    
    with col4:
        if st.button("Next ▶️", disabled=st.session_state.current_page == total_pages):
            st.session_state.current_page += 1
            st.rerun()
    
    # Calculate start and end indices
    start_idx = (st.session_state.current_page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    
    # Display paginated data
    st.dataframe(engine.df.iloc[start_idx:end_idx], use_container_width=True)

if __name__ == "__main__":
    main()