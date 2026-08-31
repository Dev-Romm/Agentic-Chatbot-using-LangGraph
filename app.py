from backend import chatbot, get_all_threads, ingest_rag_document
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import streamlit as st
import tempfile
import uuid

# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())

# Add a new thread ID to the conversation list
def add_thread(thread_id):
    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# Create a completely new chat conversation
def reset_chat():
    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])

# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):
    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])

def get_thread_label(thread_id):
    for message in load_conversation(thread_id):
        if isinstance(message, HumanMessage):
            content = str(message.content).strip().replace("\n", " ")
            return content[:40] + ("..." if len(content) > 40 else "")

    return "New conversation"


def stream_assistant_response(user_input, config, tool_status):
    """Yield answer text while reporting tool activity in the UI."""
    tool_name = None
    tool_started = False

    for message_chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages"
    ):
        tool_calls = getattr(message_chunk, "tool_calls", [])
        tool_call_chunks = getattr(message_chunk, "tool_call_chunks", [])

        if tool_calls or tool_call_chunks:
            calls = tool_calls or tool_call_chunks
            tool_name = calls[0].get("name") or "tool"
            tool_status.update(label=f"Using {tool_name}...", state="running")
            tool_started = True

        if isinstance(message_chunk, ToolMessage):
            completed_name = tool_name or "tool"
            tool_status.update(label=f"Finished {completed_name}", state="complete")

        if isinstance(message_chunk, AIMessage) and message_chunk.content:
            yield message_chunk.content

    if tool_started:
        tool_status.update(label="Response complete", state="complete")

# Display the main application title
st.title("Agentic Chatbot with LangGraph")

# create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state or st.session_state["chat_threads"] is None:
    st.session_state["chat_threads"] = get_all_threads() or []

# Add the current thread to the conversation list
add_thread(st.session_state["thread_id"])


# ++++++++++++++++++++++++ Sidebar threading feature +++++++++++++++++++++++++

# Upload and index a PDF for rag_tool
st.sidebar.title("RAG Document")
uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    help="Upload a PDF to make its contents available to rag_tool.",
)

if uploaded_pdf is not None:
    upload_id = (uploaded_pdf.name, uploaded_pdf.size)
    if st.session_state.get("rag_upload_id") != upload_id:
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
                temporary_file.write(uploaded_pdf.getvalue())
                temporary_path = temporary_file.name

            ingest_rag_document(temporary_path)
            st.session_state["rag_upload_id"] = upload_id
            st.session_state["rag_document_name"] = uploaded_pdf.name
            st.sidebar.success(f"Ready: {uploaded_pdf.name}")
        except Exception as error:
            st.sidebar.error(f"Could not index PDF: {error}")
        finally:
            if "temporary_path" in locals():
                import os
                os.unlink(temporary_path)
    else:
        st.sidebar.success(f"Ready: {uploaded_pdf.name}")

# Display the sidebar title
st.sidebar.title("My Conversations")

# Create a button for starting a new conversation
if st.sidebar.button("New Chat"):
    # Reset the current state and create a new thread
    reset_chat()

    # Rerun the streamlit app to update the interface
    st.rerun()


# Display all conversation threads in reverse order
# This shows the newest conversation first
for thread_id in st.session_state["chat_threads"][::-1]:
    # Create one sidebar button for every conversation
    if st.sidebar.button(
        get_thread_label(thread_id),
        key=thread_id
    ):
        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages into Streamlist's required message format
        temp_messages = []

        # Loop through all saved messages
        for message in messages:
            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"
            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"
            # Ignore other message types, such as ToolMessage
            else:
                continue

            # Convert the LangChain message into a dictionary
            temp_messages.append({
                "role": role,
                "content": message.content
            })

        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # Rerun the application to display the loaded messages
        st.rerun()



# ++++++++++++++++++++++++ Main Chat Interface ++++++++++++++++++++++++++++

# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# Create the chat input box
user_input = st.chat_input("Type here")

# Run this block after the user submits a message
if user_input:
    # Save the user's message in streamlit session state
    st.session_state["message_history"].append({"role":"user", "content":user_input})

    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_input)

    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name":"chat_trace",
    }

    # Create the assistant chat message container
    with st.chat_message("assistant"):
        tool_status = st.status("Thinking...", expanded=True)
        ai_message = st.write_stream(
            stream_assistant_response(user_input, CONFIG, tool_status)
        )
        tool_status.update(label="Complete", state="complete", expanded=False)

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})