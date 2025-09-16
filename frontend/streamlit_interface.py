"""
Knexion Frontend - Streamlit Chat Interface

This module provides a Streamlit-based web interface for the Knexion Agentic Knowledge 
Orchestrator. Users can upload PDF documents to create knowledge bases and interact 
with them through a conversational interface.

Features:
- PDF document upload and processing
- Multi-threaded conversation management
- Knowledge graph visualization
- Chat history persistence
"""

import streamlit as st
import uuid
import requests
from typing import List, Dict, Optional
import streamlit.components.v1 as components

# Backend API configuration
API_URL = "http://localhost:8000"

def fetch_json(endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
    """
    Make a GET request to the backend API and return JSON response.
    
    Args:
        endpoint: API endpoint path (e.g., "/get-threads")
        params: Optional query parameters
    
    Returns:
        JSON response as dict, or None if request failed
    """
    try:
        response = requests.get(f"{API_URL}{endpoint}", params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend request failed: {e}")
        return None


class ChatApp:
    """
    Main chat application class managing the Streamlit UI and state.
    
    This class handles:
    - Session state initialization and management
    - Thread/conversation management
    - File upload and processing
    - Chat interface rendering
    - Knowledge graph visualization
    """
    
    def __init__(self):
        """Initialize the chat application and session state."""
        self.initialize_session_state()

    def initialize_session_state(self) -> None:
        """
        Initialize Streamlit session state variables.
        
        Sets up:
        - message_history: List of chat messages for current thread
        - thread_id: Current conversation thread identifier
        - chat_threads: List of all available conversation threads
        """
        if 'message_history' not in st.session_state:
            st.session_state['message_history'] = []
        
        if 'thread_id' not in st.session_state:
            st.session_state['thread_id'] = None
        
        if 'chat_threads' not in st.session_state:
            # Fetch existing threads from backend
            all_threads = fetch_json("/get-threads")
            if all_threads is not None:
                st.session_state['chat_threads'] = all_threads
            else:
                st.error("Problem connecting to backend")

    @staticmethod
    def generate_thread_id() -> uuid.UUID:
        """Generate a unique thread identifier for new conversations."""
        return uuid.uuid4()

    def add_thread(self, thread_id: uuid.UUID) -> None:
        """
        Add a new thread to the session state if it doesn't exist.
        
        Args:
            thread_id: UUID of the thread to add
        """
        if thread_id not in st.session_state['chat_threads']:
            st.session_state['chat_threads'].append(thread_id)

    def reset_chat(self) -> None:
        """Reset current chat session and prepare for new conversation."""
        st.session_state['thread_id'] = None
        st.session_state['message_history'] = []

    def load_conversation(self, thread_id: uuid.UUID) -> List[Dict[str, str]]:
        """
        Load conversation history for a specific thread from backend.
        
        Args:
            thread_id: UUID of the thread to load
        
        Returns:
            List of message dictionaries with role, content, and metadata
        """
        try:
            msg_history = fetch_json("/get-msg-history", params={'thread_id': thread_id})
            return msg_history if msg_history else []
        except Exception as e:
            st.error(f"Error loading conversation: {e}")
            return []

    def switch_conversation(self, thread_id: uuid.UUID) -> None:
        """
        Switch to a different conversation thread.
        
        Args:
            thread_id: UUID of the thread to switch to
        """
        st.session_state['thread_id'] = thread_id
        st.session_state['message_history'] = self.load_conversation(thread_id)

    def render_sidebar(self) -> None:
        """
        Render the sidebar with conversation management controls.
        
        Displays:
        - New chat button
        - List of existing conversation threads
        - Thread selection interface
        """
        st.sidebar.title('Knexion Chat')
        
        # New chat button
        if st.sidebar.button('New Chat', type='primary'):
            self.reset_chat()
            st.rerun()

        # Conversation history
        st.sidebar.header('Conversations')
        
        # Display threads in reverse order (most recent first)
        for thread_id in reversed(st.session_state['chat_threads']):
            is_current = thread_id == st.session_state['thread_id']
            
            # Visual indicator for current thread
            label = f"{'🔵 ' if is_current else ''}Chat {str(thread_id)[:8]}..."
            
            if st.sidebar.button(label, key=f"thread_{thread_id}"):
                if not is_current:
                    self.switch_conversation(thread_id)
                    st.rerun()

    def render_file_upload(self) -> None:
        """
        Render the file upload interface for PDF documents.
        
        Handles:
        - Multiple PDF file selection
        - File processing via backend API
        - Thread creation for new knowledge base
        - Success/error feedback
        """
        st.header("📁 Upload Documents")
        st.write("Upload PDF files to create your knowledge base and start chatting.")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to create a knowledge base for the chatbot."
        )

        if uploaded_files:
            with st.spinner("Processing files... This may take a moment."):
                # Prepare files for upload
                files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                thread_id = self.generate_thread_id()
                data = {"thread_id": thread_id}
                
                try:
                    # Send files to backend for processing
                    response = requests.post(f"{API_URL}/upload", files=files, data=data)
                    response.raise_for_status()
                    
                    result_data = response.json()
                    
                    if result_data['result'] == "successful":
                        st.success("Files processed successfully!")
                        
                        # Set up new conversation thread
                        st.session_state['thread_id'] = thread_id
                        st.session_state['message_history'] = []
                        self.add_thread(thread_id)
                        
                        st.rerun()
                    else:
                        st.error("Error processing files")
                        
                except Exception as e:
                    st.error(f"Error processing files: {e}")

    @st.dialog("Context", width="medium")
    def render_context_dialog(self, kg_path: Optional[str] = None, docs: Optional[str] = None):
        """
        Render a dialog showing knowledge graph and document context.
        
        Args:
            kg_path: Path to knowledge graph HTML visualization
            docs: Retrieved document context as string
        """
        tab1, tab2 = st.tabs(["Knowledge Graph", "Documents"])
        
        with tab1:
            st.write("Knowledge Graph Visualization")
            if kg_path:
                try:
                    response = requests.get(API_URL + "/get-kg-html", params={"filename": kg_path})
                    if response.status_code == 200:
                        components.html(response.text, height=600, width=800, scrolling=True)
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown Error')}")
                except Exception:
                    st.error("Couldn't load graph at this time!")
            else:
                st.write("No knowledge graph was retrieved for this response")

        with tab2:
            st.write("Retrieved Document Context")
            if docs:
                st.text_area("Documents", value=docs, height=400, disabled=True)
            else:
                st.write("No documents were retrieved")

        if st.button("Close"):
            st.rerun()

    def render_chat_message(self, role: str, content: str, kg_path: Optional[str] = None, 
                          docs: Optional[str] = None) -> None:
        """
        Render a single chat message with optional context button.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content text
            kg_path: Optional path to knowledge graph visualization
            docs: Optional document context
        """
        with st.chat_message(role):
            st.write(content)
            
            # Add context button for assistant messages
            if role == "assistant" and (kg_path or docs):
                if st.button("View Context", key=f"context_{kg_path}_{hash(content)}"):
                    self.render_context_dialog(kg_path, docs)

    def handle_user_input(self, user_input: str) -> None:
        """
        Process user input and generate AI response.
        
        Args:
            user_input: User's question or message
        
        Flow:
        1. Add user message to history
        2. Display user message
        3. Show "thinking..." placeholder
        4. Query backend for AI response
        5. Update with actual response and context
        """
        # Add user message to history and display
        user_message = {
            'role': 'user',
            'content': user_input,
            'kg_path': None,
            'docs': None
        }
        st.session_state['message_history'].append(user_message)
        self.render_chat_message('user', user_input)

        # Show thinking placeholder
        thinking_placeholder = {
            'role': 'assistant',
            'content': "Thinking...",
            'kg_path': None,
            'docs': None
        }
        st.session_state['message_history'].append(thinking_placeholder)
        self.render_chat_message('assistant', "Thinking...")

        # Get AI response from backend
        ai_response = fetch_json("/query", params={
            "q": user_input, 
            "thread_id": st.session_state['thread_id']
        })

        if ai_response:
            # Replace thinking placeholder with actual response
            ai_message = {
                'role': 'assistant',
                'content': ai_response['answer'],
                'kg_path': ai_response['kg_path'],
                'docs': ai_response['docs']
            }
            st.session_state['message_history'][-1] = ai_message
            st.rerun()

    def render_chat_interface(self) -> None:
        """
        Render the main chat interface with message history and input.
        
        Displays:
        - Chat message history
        - Chat input field
        - Message processing
        """
        st.header("💬 Chat Interface")

        # Display conversation history
        for message in st.session_state['message_history']:
            self.render_chat_message(
                message['role'], 
                message['content'], 
                message.get("kg_path"), 
                message.get("docs")
            )

        # Chat input field
        user_input = st.chat_input("Ask a question about your documents...")
        if user_input:
            self.handle_user_input(user_input)

    def run(self) -> None:
        """
        Main application entry point.
        
        Sets up the Streamlit page configuration and renders the appropriate
        interface based on current state (file upload or chat).
        """
        st.set_page_config(
            page_title="Knexion - Agentic Knowledge Orchestrator",
            page_icon="🔗",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Always render sidebar
        self.render_sidebar()

        # Main content: show chat interface if thread exists, otherwise show upload
        if st.session_state['thread_id']:
            self.render_chat_interface()
        else:
            self.render_file_upload()


def main():
    """Application entry point - creates and runs the ChatApp."""
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()