# 🤖 Agentic Chatbot with LangGraph

A comprehensive chatbot framework built with **LangGraph** and **Azure OpenAI**, featuring multiple AI agent configurations with increasing complexity—from simple conversation to intelligent agents with tools, RAG, and persistent memory.

## 📋 Project Overview

A unified, intelligent chatbot built with **LangGraph** and **Azure OpenAI**, featuring:

- **Multi-tool Integration** - Web search, weather, calculator, and document Q&A
- **Retrieval-Augmented Generation (RAG)** - Query uploaded PDF documents
- **Persistent Memory** - SQLite-based conversation history with thread support
- **Agentic Capabilities** - LLM dynamically selects and uses appropriate tools
- **Interactive UI** - Streamlit web interface

The chatbot uses **Azure OpenAI GPT** as the LLM backbone and **LangGraph** for intelligent workflow orchestration.

---

## 🏗️ Project Structure

```
.
├── app.py                          # Streamlit UI application
├── backend.py                      # Core chatbot backend logic
├── requirements.txt                # Project dependencies
├── Dockerfile                      # Docker container configuration
├── .dockerignore                   # Files excluded from Docker build
├── .gitignore                      # Files excluded from version control
├── README.md                       # Project documentation
├── .env                            # Environment variables (not in version control)
└── .github/                        # GitHub configuration and workflows
```

---

## 🚀 Features

### **Multi-Tool Intelligence**
The chatbot dynamically selects the best tool(s) for each query:
- **🔍 Web Search** (Tavily API) - Current events, recommendations, real-world information
- **📄 RAG (Document Q&A)** - Query uploaded PDF documents with semantic search (FAISS)
- **🧮 Calculator** - Mathematical expressions and computations
- **🌤️ Weather** (Weatherstack API) - Real-time weather for any location

### **Smart Routing**
- LLM automatically decides which tool(s) to use for each query
- Forces live search for time-sensitive queries ("latest", "current", "today")
- Falls back to direct response when no tool is needed

### **Persistent Memory**
- SQLite database stores complete conversation history
- Resume conversations across sessions using thread IDs
- Thread-safe concurrent access

### **Document Analysis**
- Ingest PDF documents into FAISS vector store
- Semantic search for relevant passages
- Source and page references in responses

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Azure OpenAI API credentials
- Tavily API key (for search capabilities)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agentic-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # Azure OpenAI Configuration
   AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
   AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
   
   # Tavily Search API (for tools chatbot)
   TAVILY_API_KEY=<your-tavily-api-key>
   ```

---

## 🎯 Usage

### Run the Chatbot
```bash
streamlit run app.py
```

The application opens in your browser at `http://localhost:8501`

### Load Documents for RAG
Within the Streamlit UI, use the file uploader to add PDF documents. The chatbot will:
1. Extract text from the PDF
2. Split into semantic chunks
3. Create vector embeddings
4. Store in FAISS for semantic search

### Example Queries
- **Direct question**: "What is machine learning?"
- **Search-based**: "What are the latest developments in AI?"
- **Calculation**: "What is 25% of 1200?"
- **Weather**: "What's the weather in London?"
- **Document Q&A**: "Summarize the findings from the uploaded PDF"

---

## 🔧 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Azure OpenAI GPT-4 | Language understanding & generation |
| **Workflow Engine** | LangGraph | Agent orchestration & state management |
| **Vector Store** | FAISS | Semantic search for RAG |
| **Document Loading** | PyPDF | PDF ingestion for RAG |
| **Web Search** | Tavily API | Real-time information retrieval |
| **UI Framework** | Streamlit | Interactive web interface |
| **Persistence** | SQLite | Conversation history storage |
| **Embeddings** | Azure OpenAI Embeddings | Vector representations |

---

## 📚 Core Components

### ChatState
```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```
Manages conversation state using LangGraph's message aggregation.

### Persistence Layer
- **SQLite**: `SqliteSaver(conn)` - Persistent conversation storage
- **Thread IDs**: Unique identifiers for resumable conversations
- **Database**: `chatbot.db` (auto-created)

### Workflow Architecture
```
START → chat_node (LLM decides) → tools (if needed) → chat_node → END
         ↓                                                ↑
         └────────────────────────────────────────────────┘
         Conditional routing based on tool requirements
```

### Available Tools
1. **search_tool** - Tavily web search
2. **get_current_weather** - Weatherstack API
3. **calculator** - Math evaluation
4. **rag_tool** - FAISS vector store retrieval

---

## 🧠 RAG (Retrieval-Augmented Generation)

### Document Processing Pipeline
1. **Load** - Extract text from PDF using PyPDFLoader
2. **Split** - Chunk into semantic pieces (1000 chars, 200 char overlap)
3. **Embed** - Create vector embeddings using Azure OpenAI Embeddings
4. **Store** - Save to FAISS vector database (`faiss_db/`)

### Retrieval Process
1. User query is embedded using same model
2. Semantic similarity search (top-4 chunks)
3. Retrieved passages passed as context to LLM
4. LLM generates response with source references

### Configuration
```python
# In backend.py
RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust for doc complexity
    chunk_overlap=200     # Adjust for context continuity
)
```

---

## 🛠️ Tool Integration & Agent Flow

The chatbot implements a ReAct-style agent pattern with intelligent tool selection:

### Tool Decision Logic
- **Forced Search**: Triggers for time-sensitive queries ("latest", "current", "upcoming")
- **Optional Tools**: LLM chooses best tool(s) for other queries
- **Tool Composition**: Multiple tools can be used in a single response

### Execution Flow
```
User Query
    ↓
LLM with Tools (or Required Tools)
    ↓
Tool Call? (conditional routing)
    ├─ Yes → Execute tool(s) → Return results
    │        ↓
    │    LLM generates response with context
    └─ No → LLM responds directly
    ↓
Return response to user
```

### Tool Configuration
- **Search**: Tavily API (max 5 results, advanced depth)
- **Weather**: Weatherstack API (real-time conditions)
- **Calculator**: Safe eval with math library
- **RAG**: FAISS with semantic similarity

---

## 📖 Development & Customization

### Extending the Backend
```python
# Add new tools in backend.py
@tool
def my_tool(param: str) -> str:
    """Tool description and usage."""
    # Implementation
    return result

# Add to tools list
tools = [search_tool, get_current_weather, calculator, rag_tool, my_tool]
```

### Customizing LLM Behavior
Edit the `system_message` in `chat_node()` to:
- Change tool usage guidelines
- Adjust response tone and style
- Add domain-specific instructions

### Vector Store Management
```python
from backend import ingest_rag_document, get_retriever

# Ingest a new document
ingest_rag_document("path/to/document.pdf")

# Retrieve documents programmatically
retriever = get_retriever()
docs = retriever.invoke("search query")
```

---

## 🔒 Security & Best Practices

- **API Keys**: Store in `.env` file, never commit to version control
- **Database**: SQLite stored locally; for production consider PostgreSQL
- **Rate Limiting**: Tavily API has rate limits; implement backoff strategies
- **Context Length**: Monitor LLM token usage for long conversations

---

## 📊 Configuration

### Azure OpenAI Deployments
Update the deployment names in `backend.py`:
```python
llm = AzureChatOpenAI(
    deployment_name="gpt-5-mini",  # Your deployment name
    openai_api_version="2025-04-01-preview"
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_deployment="text-embedding-3-small",
    api_version="2023-05-15"
)
```

### Required Environment Variables
Add these to your `.env` file:
```env
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=<your-endpoint>
TAVILY_API_KEY=<your-tavily-key>
WEATHERSTACK_API_KEY=<your-weatherstack-key>
```

### RAG Settings
Customize document processing in `backend.py`:
```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Adjust for document type
    chunk_overlap=200       # Adjust for semantic continuity
)
```

### Tool Configuration
Modify tool parameters in `backend.py`:
```python
search_tool = TavilySearch(
    max_results=5,          # Number of search results
    topic="general",        # Topic focus
    search_depth="advanced" # Search depth level
)
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Azure OpenAI connection fails | Verify API key and endpoint in `.env` |
| Streamlit not starting | Ensure `.venv` is activated and dependencies installed |
| FAISS index not found | Run document ingestion first via `ingest_rag_document()` |
| Tools not responding | Check Tavily API key and rate limits |
| SQLite database locked | Ensure only one Streamlit instance is running |

---

## 🎓 Learning Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Azure OpenAI API Docs](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

## 📝 License

This project is provided as-is for educational and development purposes.

---

## 🤝 Contributing

Feel free to extend this project with:
- Additional tools (email, calendar, database queries)
- Memory augmentation (semantic memory, long-term context)
- Multi-agent systems
- Different LLM providers
- Advanced RAG techniques (reranking, hybrid search)

---

## 📞 Support

For issues or questions, refer to:
1. Individual backend file docstrings
2. Jupyter notebook examples
3. LangGraph and LangChain communities

---

**Happy coding! 🚀**
