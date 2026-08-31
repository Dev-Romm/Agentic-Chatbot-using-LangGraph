# 🤖 Agentic Chatbot with LangGraph

A comprehensive chatbot framework built with **LangGraph** and **Azure OpenAI**, featuring multiple AI agent configurations with increasing complexity—from simple conversation to intelligent agents with tools, RAG, and persistent memory.

## 📋 Project Overview

This project demonstrates different chatbot architectures using LangGraph's state management and workflow capabilities:

- **Basic Chatbot** - Simple conversational AI with memory
- **Database Chatbot** - Persistent conversation history with SQLite
- **RAG Chatbot** - Retrieval-Augmented Generation for document-based Q&A
- **Tools Chatbot** - Agentic capabilities with external tools (search, calculator, weather)

All implementations use **Streamlit** for an interactive web UI and **Azure OpenAI** as the LLM backbone.

---

## 🏗️ Project Structure

```
agentic_chatbot_backend.py          # Basic chatbot implementation
agentic_chatbot_db_backend.py       # Database-persistent chatbot backend
agentic_chatbot_rag_backend.py      # RAG-enabled chatbot backend
agentic_chatbot_tools_backend.py    # Tools-enabled agentic backend

app.py                              # Basic chatbot Streamlit UI
app_db.py                           # Database chatbot Streamlit UI
app_rag.py                          # RAG chatbot Streamlit UI
app_tools.py                        # Tools chatbot Streamlit UI

app_thread.py                       # Thread management utilities
test.py                             # Testing utilities

requirements.txt                    # Project dependencies
chatbot.db                          # SQLite database (generated)

faiss_db/                           # FAISS vector store index
notebooks/
  ├── rag_demo.ipynb               # RAG implementation demonstration
  └── chatbot_workflow.ipynb       # General chatbot workflow notebook
```

---

## 🚀 Features

### 1. **Basic Chatbot** (`app.py`)
- Simple conversational interface
- In-memory message history
- Powered by Azure OpenAI GPT-4
- Thread-based conversation state management

### 2. **Database Chatbot** (`app_db.py`)
- Persistent conversation storage with SQLite
- Load and resume previous conversations
- Multiple independent chat threads
- Conversation history management

### 3. **RAG Chatbot** (`app_rag.py`)
- Document ingestion and indexing (PDF support)
- FAISS vector store for semantic search
- Context-aware responses based on uploaded documents
- Combines retrieval with generation

### 4. **Tools Chatbot** (`app_tools.py`)
- Multi-tool integration:
  - **Web Search** (Tavily API)
  - **Calculator** (Math expressions)
  - **Weather API** (Real-time weather data)
- Agentic decision-making
- Tool-use patterns with LangGraph

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

### Run Basic Chatbot
```bash
streamlit run app.py
```

### Run Database Chatbot (with history)
```bash
streamlit run app_db.py
```

### Run RAG Chatbot (document QA)
```bash
streamlit run app_rag.py
```

### Run Tools Chatbot (agentic with tools)
```bash
streamlit run app_tools.py
```

Each application opens in your browser at `http://localhost:8501`

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

### ChatState (Shared Across All Implementations)
```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```
Manages conversation state with automatic message aggregation.

### Checkpointing Options
- **Memory**: `MemorySaver()` - In-memory persistence (basic chatbot)
- **SQLite**: `SqliteSaver(conn)` - Persistent database storage (DB & Tools chatbots)

### LangGraph Workflow Pattern
```
START → Chat/Agent Node → Tool Node (if applicable) → END
```

---

## 🧠 RAG (Retrieval-Augmented Generation)

### Document Ingestion
```python
def ingest_rag_document(file_path):
    # Loads PDF → Splits into chunks → Creates embeddings → Stores in FAISS
```

### Retrieval Process
1. User query is embedded
2. Semantic similarity search in FAISS
3. Top documents retrieved as context
4. LLM generates response with context

---

## 🛠️ Tools Integration (Agentic Chatbot)

The tools chatbot implements a ReAct-style agent pattern:

**Available Tools:**
- `search_tool`: Web search via Tavily
- `calculator`: Mathematical expression evaluation
- `get_current_weather`: Real-time weather data

**Agent Flow:**
1. LLM receives user query
2. Model decides which tool(s) to use
3. Tools execute and return results
4. LLM generates final response with tool context

---

## 📖 Development & Testing

### Run Tests
```bash
python test.py
```

### Jupyter Notebooks
- `notebooks/chatbot_workflow.ipynb` - Workflow demonstrations
- `notebooks/rag_demo.ipynb` - RAG implementation details

### Thread Management
Use `app_thread.py` for manual thread operations:
```python
from app_thread import generate_thread_id, load_conversation
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
Update the deployment names in backend files if using different models:
```python
llm = AzureChatOpenAI(
    deployment_name="gpt-41-mini",  # Your deployment name
    openai_api_version="2024-02-15-preview"
)
```

### RAG Settings
Customize chunking in `agentic_chatbot_rag_backend.py`:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Adjust chunk size
    chunk_overlap=200       # Adjust overlap
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
