import streamlit as st
from vectorDB_manager import ChromaDBHandler
from tokenizer import TextTokenizer
from utils import config, logger

# Configure Streamlit page
st.set_page_config(page_title="RAG Agent", layout="wide")

# Initialize session state
if "hf_token" not in st.session_state:
    st.session_state["hf_token"] = config["hugging_face"]["token"]
if "hf_model" not in st.session_state:
    st.session_state["hf_model"] = config["hugging_face"]["model"]
if "collection_name" not in st.session_state:
    st.session_state["collection_name"] = ""
if "db_folder" not in st.session_state:
    st.session_state["db_folder"] = ""
if "use_existing_db" not in st.session_state:
    st.session_state["use_existing_db"] = True
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def clear_chat():
    """Clear chat history."""
    if st.button("Yes, clear", key="confirm_clear"):
        st.session_state["chat_history"] = []
        st.success("Chat history cleared!")
    else:
        st.warning("Chat not cleared.")

with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat"):
        st.warning("Are you sure you want to clear the chat?")
        clear_chat()


# Hugging Face Authentication Screen
st.title("RAG Agent Setup")

# Input Hugging Face token and model
if (st.session_state["hf_token"] is None) or (st.session_state["hf_model"] is None):
    with st.form("hf_auth_form"):
        st.subheader("Enter Hugging Face Credentials")
        if not st.session_state["hf_token"]:
            hf_token = st.text_input("Hugging Face Token", type="password")
        if not st.session_state["hf_model"]:
            hf_model = st.text_input("Model Name (e.g., 'gpt-neo', 'distilbert')")
        submitted = st.form_submit_button("Save and Proceed")

        if submitted:
            st.session_state["hf_token"] = hf_token
            st.session_state["hf_model"] = hf_model
            st.success("Credentials saved! Proceed to database setup.")

def create_client(folder):
    st.session_state['chroma_client'] = ChromaDBHandler(folder)

# Database Setup
if st.session_state["hf_token"] and st.session_state["hf_model"]:
    st.subheader("Database Setup")

    # Choose between existing DB or creating a new one
    db_option = st.radio(
        "Choose Database Option",
        ("Use Existing VectorDB", "Create New VectorDB"),
        index=0
    )
    st.session_state["use_existing_db"] = db_option == "Use Existing VectorDB"

    if st.session_state["use_existing_db"]:
        folder = st.text_input("Enter folder path for existing VectorDB collections",on_change=create_client)
        if folder:
            collections = st.session_state['chroma_client'].list_chromadb_collections()
            collection = st.selectbox("Choose a collection", [i.name for i in collections])
        if st.button("Load Collection"):
            try:
                st.session_state['chroma_client'].load_existing_collection(collection)
                st.session_state["collection_name"] = collection
                st.session_state["db_folder"] = folder
                st.success(f"Selected Collection: {collection}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        folder = st.text_input("Enter folder path to save the new VectorDB")
        new_collection_name = st.text_input("Enter new collection name")
        text_folder = st.text_input("Enter folder path containing text documents")
        file_mode = st.selectbox("Choose all files in a folder or only parent folder", ["Recursive", "Non-recursive"])
        
        if folder and new_collection_name and text_folder and file_mode:
            st.session_state["collection_name"] = new_collection_name
            st.session_state["db_folder"] = folder
            if st.button("Create Collection"):
                try:
                    st.session_state['chroma_client'] = ChromaDBHandler(folder)
                    st.session_state['chroma_client'].create_or_replace_collection(new_collection_name)
                except Exception as e:
                    st.error(f"Error: {e}")
                st.session_state['tokeniser'] = TextTokenizer(file_mode=file_mode, folder_path=text_folder)
                st.session_state['text_tokens'] = st.session_state['tokeniser'].text
                st.session_state['text_metadata'] = st.session_state['tokeniser'].metadata
                st.session_state['chroma_client'].add_tokens(st.session_state['text_tokens'] ,st.session_state['text_metadata'])
                st.success(f"New collection '{new_collection_name}' created at {folder}!")

        
# Chat Interface
if st.session_state["collection_name"]:
    st.subheader("Chat Interface")

    # Main container for chat
    with st.container():
        # Input and response section
        try:
            user_query = st.text_input("Enter your query", key="user_query")
            if st.button("Get Response"):
                query = st.session_state.get("user_query", "")
                top_context = st.session_state['chroma_client'].retrieve_top_n(query)
                if query:
                    # Replace with your RAG agent integration logic
                    response = f"Response to '{query}' from model '{st.session_state.get('hf_model', 'default model')}': {top_context}"
                    
                    # Append the query and response to chat history
                    st.session_state["chat_history"].append({"user": query, "model": response})
                else:
                    st.warning("Please enter a query.")

            # Display chat history
            if st.session_state["chat_history"]:
                st.markdown("### Chat History")
                for chat in st.session_state["chat_history"][::-1]:
                    st.markdown(f"**User:** {chat['user']}")
                    st.markdown(f"**Model:** {chat['model']}")
        except Exception as e:
            st.error(f"Error: {e}")
