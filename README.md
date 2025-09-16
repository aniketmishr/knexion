# Knexion: Agentic Knowledge Orchestrator

**Connecting concepts, bridging knowledge gaps, delivering trusted answers.**

Knexion is an AI-powered knowledge orchestration system that transforms static PDF documents into a living, adaptive knowledge network. By combining knowledge graphs, vector embeddings, and real-time web intelligence, it provides fact-grounded, contextual answers that adapt and improve with every interaction.

## 🎥 Project Demo

**[Watch the Complete Project Walkthrough](YOUR_YOUTUBE_LINK_HERE)**

*Click above to see Knexion in action - from document upload to intelligent question answering with knowledge graph visualization.*

---

## 🚀 What Makes Knexion Special

Unlike traditional RAG systems that simply retrieve documents, Knexion creates an interconnected web of knowledge that:

- **Structures Knowledge**: Converts PDFs into knowledge graphs with entities and relationships
- **Bridges Gaps**: Autonomously searches the web when local knowledge is insufficient  
- **Validates Answers**: Multi-layer verification prevents hallucinations and ensures accuracy
- **Visualizes Connections**: Interactive knowledge graphs show how concepts relate
- **Learns Continuously**: Each interaction enriches the knowledge base

## 🏗️ Architecture Overview

```
📄 PDF Upload → 🧠 Knowledge Graph + 🔍 Vector Embeddings → 💬 Hybrid Retrieval → 🤖 Agent Workflow → ✅ Validated Response
```

### Core Components

- **Knowledge Graph**: Entities and relationships extracted using LLM
- **Vector Store**: Semantic embeddings for similarity search
- **Agent Workflow**: LangGraph orchestration with quality gates
- **Web Integration**: Tavily search for knowledge gap filling
- **Validation Pipeline**: Hallucination detection and answer grading

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Backend API** | FastAPI |
| **Database** | TiDB (Vector + Graph storage) |
| **LLM** | Google Gemini 2.0 Flash |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Workflow** | LangGraph |
| **Web Search** | Tavily API |
| **Visualization** | PyVis |

## 📁 Project Structure

```
knexion/
├── streamlit_interface.py    # Streamlit UI components and chat interface
├── api_server.py            # FastAPI backend with REST endpoints
├── workflow_orchestrator.py # LangGraph agent workflow and routing
├── llm_chains.py           # LLM evaluation chains and grading
├── knowledge_store.py      # TiDB operations for KG and vectors
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- TiDB Cloud account
- OpenAI API key
- Tavily API key
- Google AI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/knexion.git
   cd knexion
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database credentials
   ```

4. **Create cache directory**
   ```bash
   mkdir -p .cache/graph
   ```

### Environment Variables

Create a `.env` file with the following variables:

```env
# TiDB Configuration
TIDB_CONNECTION_STRING=mysql+pymysql://username:password@host:port/database
TIDB_HOST_NAME=your-tidb-host
TIDB_PORT_NUMBER=4000
TIDB_USERNAME=your-username
TIDB_PASSWORD=your-password
TIDB_DATABASE_NAME=your-database

# API Keys
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-ai-api-key
TAVILY_API_KEY=your-tavily-api-key
```

### Running the Application

1. **Start the backend server**
   ```bash
   python api_server.py
   ```

2. **Launch the frontend** (in a new terminal)
   ```bash
   streamlit run streamlit_interface.py
   ```

3. **Open your browser**
   Navigate to `http://localhost:8501`

## 💡 How to Use

### 1. Upload Documents
- Click "Choose PDF files" to upload your documents
- Wait for processing (knowledge graph creation + vector embedding)
- System creates a new conversation thread automatically

### 2. Ask Questions
- Type questions about your uploaded documents
- Get answers enriched with both structured knowledge and semantic context
- View knowledge graph visualizations by clicking "View Context"

### 3. Explore Knowledge
- Interactive knowledge graphs show entity relationships
- Document context reveals source material for transparency
- Thread management allows multiple conversation histories

## 🔧 Advanced Configuration

### Chunk Size Settings
Modify in `api_server.py`:
```python
CHUNK_SIZE = 1000      # Adjust based on document complexity
CHUNK_OVERLAP = 200    # Overlap for context preservation
```

### Knowledge Graph Depth
Modify in `knowledge_store.py`:
```python
def retrieve_knowledge_graph(query, metadata, max_depth=3, top_k=10):
    # Adjust max_depth for broader/narrower context
```

### Model Configuration
Modify in `llm_chains.py`:
```python
llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0  # Adjust for creativity vs consistency
)
```

## 🔍 Agent Workflow

The system follows this intelligent decision process:

1. **Retrieve**: Get context from knowledge graph + vector search
2. **Grade Documents**: Evaluate relevance of retrieved content
3. **Generate**: Create answer using hybrid context
4. **Validate**: Check for hallucinations and answer quality
5. **Web Search**: Fetch external knowledge if gaps detected
6. **Iterate**: Regenerate with enriched context until satisfied

## 🎯 Key Features

### Hybrid Retrieval System
- **Semantic Search**: Vector similarity for document passages
- **Graph Reasoning**: Entity relationships for structured knowledge
- **Dynamic Fusion**: Intelligent combination of both approaches

### Quality Assurance Pipeline
- **Relevance Grading**: Filter irrelevant retrieved documents
- **Hallucination Detection**: Ensure answers are grounded in facts
- **Answer Quality**: Verify responses address user questions
- **Iterative Improvement**: Regenerate until quality thresholds met

### Knowledge Gap Bridging
- **Automatic Detection**: Identify when local knowledge is insufficient
- **Web Search Integration**: Fetch external information seamlessly
- **Context Augmentation**: Enrich responses with real-time data

## 📊 Performance Considerations

- **Vector Search**: Optimized with TiDB's native vector operations
- **Graph Traversal**: BFS algorithm with configurable depth limits
- **Caching**: Knowledge graph visualizations cached for reuse
- **Batch Processing**: Efficient document chunking and embedding

## 🛡️ Security & Privacy

- **API Key Management**: Secure environment variable handling
- **File Validation**: PDF type checking and size limits
- **Path Security**: Directory traversal protection
- **Data Isolation**: Thread-based data separation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TiDB** for vector database capabilities
- **LangGraph** for workflow orchestration
- **Streamlit** for rapid UI development
- **OpenAI** for embedding models
- **Google AI** for language models
- **Tavily** for web search integration

## 📞 Support

- Create an issue for bugs or feature requests
- Check existing issues before creating new ones
- Provide detailed information for faster resolution

---

**Built with ❤️ for the future of knowledge work**

*Transform your documents into intelligent, interconnected knowledge that grows smarter with every question.*