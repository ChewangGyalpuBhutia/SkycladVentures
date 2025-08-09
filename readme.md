# Data Query & Analysis Platform

This project provides an interactive platform for querying, analyzing, and visualizing your own datasets using natural language and AI-powered chat (RAG). It supports CSV, Excel, and JSON files.



## Features

- **Upload Your Own Dataset:** No default data—upload your own file to get started.
- **Natural Language Query:** Ask questions about your data in plain English.
- **AI Chat (RAG):** Chat with your data using AI (requires GEMINI_API_KEY in `.env`).
- **Data Overview:** Get statistics, column info, and sample data.
- **Download Results:** Export query results as CSV.
- **Visualization:** Visualize your data interactively (via Streamlit).

## FastAPI Backend

A FastAPI backend is implemented in `src/main.py`, providing a REST API for data querying, AI chat (RAG), and dataset overview endpoints.

## Sample Data & Test Queries

A sample dataset is provided as `dataset.csv` in the project root. Example queries for testing are available in `queries.txt`.

## Getting Started

### 1. Install Requirements

```sh
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file in the root directory and add your Gemini API key if you want to use RAG features:

```
GEMINI_API_KEY=your_api_key_here
```

### 3. Run the App

```sh
cd src
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Upload Your Data

- Click "Upload Your Dataset" and select a CSV, Excel, or JSON file.
- Once uploaded, you can query, chat, and visualize your data.

## Project Structure

```
src/
    app.py                # Streamlit main app
    integrated_query.py   # Query engine
    rag_engine.py         # RAG/AI chat engine
    engines/              # Query sub-engines
    ui/file_uploader.py   # File upload UI
dataset/                  # (Optional) Example datasets
```

## Notes

- No default dataset is loaded; you must upload your own file.
- For AI chat, set `GEMINI_API_KEY` in `.env`.
- All processing is local except for AI chat, which uses Gemini API.

## License

MIT License
