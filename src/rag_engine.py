import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import requests
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None

@dataclass
class ConversationTurn:
    user_query: str
    ai_response: str
    timestamp: str
    context_used: List[str]

class RAGEngine:
    def __init__(self, dataframe: pd.DataFrame, gemini_api_key: str):
        self.df = dataframe
        self.gemini_api_key = gemini_api_key
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize document store
        self.documents = []
        self.index = None
        
        # Conversation history
        self.conversation_history: List[ConversationTurn] = []
        self.max_history_length = 10  # Keep last 10 exchanges
        
        # Create embeddings for data
        self._create_document_embeddings()
        
    def _create_document_embeddings(self):
        """Create embeddings for dataset content"""
        # Create document chunks from data
        self._create_data_documents()
        
        # Generate embeddings
        embeddings = []
        for doc in self.documents:
            embedding = self.model.encode(doc.content)
            doc.embedding = embedding
            embeddings.append(embedding)
        
        # Create FAISS index
        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
            self.index.add(embeddings_array)
    
    def _create_data_documents(self):
        """Convert dataframe into searchable documents"""
        # Column descriptions
        for col in self.df.columns:
            stats_text = self._get_column_description(col)
            self.documents.append(DocumentChunk(
                content=f"Column {col}: {stats_text}",
                metadata={"type": "column_info", "column": col}
            ))
        
        # Data patterns and insights
        insights = self._generate_data_insights()
        for insight in insights:
            self.documents.append(DocumentChunk(
                content=insight["content"],
                metadata={"type": "insight", "category": insight["category"]}
            ))
        
        # Sample data descriptions
        for idx, row in self.df.head(10).iterrows():
            row_description = self._describe_row(row, idx)
            self.documents.append(DocumentChunk(
                content=row_description,
                metadata={"type": "sample_data", "row_index": idx}
            ))
    
    def _get_column_description(self, column: str) -> str:
        """Generate description for a column"""
        if column in self.df.select_dtypes(include=[np.number]).columns:
            stats = self.df[column].describe()
            return (f"Numeric column with mean {stats['mean']:.2f}, "
                   f"range from {stats['min']:.2f} to {stats['max']:.2f}, "
                   f"standard deviation {stats['std']:.2f}")
        else:
            unique_count = self.df[column].nunique()
            return f"Categorical column with {unique_count} unique values"
    
    def _generate_data_insights(self) -> List[Dict]:
        """Generate insights about the dataset"""
        insights = []
        
        # Dataset overview
        insights.append({
            "content": f"This dataset contains {len(self.df)} rows and {len(self.df.columns)} columns. "
                      f"There are {len(self.df.select_dtypes(include=[np.number]).columns)} numeric columns.",
            "category": "overview"
        })
        
        # Correlation insights
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            max_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
            max_corr = max_corr[max_corr < 1.0].iloc[0]
            max_corr_idx = corr_matrix.abs().unstack().sort_values(ascending=False).index[1]
            
            insights.append({
                "content": f"Highest correlation is {max_corr:.3f} between {max_corr_idx[0]} and {max_corr_idx[1]}",
                "category": "correlation"
            })
        
        return insights
    
    def _describe_row(self, row: pd.Series, index: int) -> str:
        """Create description for a data row"""
        numeric_vals = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if not pd.isna(row[col]):
                numeric_vals.append(f"{col}: {row[col]:.2f}")
        
        return f"Row {index}: " + ", ".join(numeric_vals)
    
    def _format_conversation_history(self, recent_turns: int = 3) -> str:
        """Format recent conversation history for context"""
        if not self.conversation_history:
            return ""
        
        # Get last few conversation turns
        recent_history = self.conversation_history[-recent_turns:]
        
        formatted = "Previous conversation:\n"
        for turn in recent_history:
            formatted += f"User: {turn.user_query}\n"
            formatted += f"Assistant: {turn.ai_response}\n\n"
        
        return formatted
    
    def _detect_follow_up_query(self, query: str) -> bool:
        """Detect if this is a follow-up query"""
        follow_up_indicators = [
            'what about', 'how about', 'and', 'also', 'additionally',
            'furthermore', 'moreover', 'can you also', 'what if',
            'in that case', 'then', 'now', 'next', 'continue',
            'follow up', 'building on', 'expanding on', 'regarding that',
            'about that', 'for that', 'with that', 'those', 'these',
            'it', 'they', 'them', 'this', 'that'
        ]
        
        query_lower = query.lower().strip()
        
        # Check for pronouns and references that suggest follow-up
        if any(indicator in query_lower for indicator in follow_up_indicators):
            return True
        
        # Check if query is very short (likely referencing previous context)
        if len(query.split()) < 5:
            return True
        
        return False
    
    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant documents for a query"""
        if self.index is None:
            return []
        
        # For follow-up queries, also consider recent conversation context
        enhanced_query = query
        if self._detect_follow_up_query(query) and self.conversation_history:
            # Add context from last conversation turn
            last_turn = self.conversation_history[-1]
            enhanced_query = f"{last_turn.user_query} {query}"
        
        # Encode query
        query_embedding = self.model.encode([enhanced_query]).astype('float32')
        
        # Search similar documents
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return relevant content
        relevant_docs = []
        for idx in indices[0]:
            if idx < len(self.documents):
                relevant_docs.append(self.documents[idx].content)
        
        return relevant_docs
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response using Gemini API with conversation history"""
        # Prepare context
        context_text = "\n".join(context)
        
        # Include conversation history for follow-up queries
        conversation_context = ""
        if self._detect_follow_up_query(query) or len(self.conversation_history) > 0:
            conversation_context = self._format_conversation_history()
        
        # Create prompt
        prompt = f"""
        {conversation_context}
        
        Context about the dataset:
        {context_text}
        
        Current User Query: {query}
        
        Based on the conversation history (if any) and the dataset context provided, please answer the user's current question. 
        If this appears to be a follow-up question, reference the previous conversation appropriately.
        If you need specific data values, suggest appropriate queries.
        Be concise and helpful.
        """
        
        # Call Gemini API
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': self.gemini_api_key
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def rag_query(self, query: str) -> Dict:
        """Complete RAG pipeline with conversation history"""
        from datetime import datetime
        
        # Retrieve relevant context
        context = self.retrieve_relevant_context(query)
        
        # Generate response
        response = self.generate_response(query, context)
        
        # Store this conversation turn
        conversation_turn = ConversationTurn(
            user_query=query,
            ai_response=response,
            timestamp=datetime.now().isoformat(),
            context_used=context
        )
        
        self.conversation_history.append(conversation_turn)
        
        # Maintain history limit
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        return {
            "query": query,
            "response": response,
            "context_used": context,
            "num_context_docs": len(context),
            "is_follow_up": self._detect_follow_up_query(query),
            "conversation_turn": len(self.conversation_history)
        }
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of conversation history"""
        return {
            "total_turns": len(self.conversation_history),
            "recent_topics": [turn.user_query for turn in self.conversation_history[-3:]],
            "conversation_active": len(self.conversation_history) > 0
        }