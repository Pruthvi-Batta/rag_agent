import streamlit as st
from pathlib import Path
import os
import chromadb
from chromadb.config import Settings

# Configure Streamlit page
st.set_page_config(page_title="RAG Agent", layout="wide")

# Initialize session state
if "hf_token" not in st.session_state:
    st.session_state["hf_token"] = ""
if "hf_model" not in st.session_state:
    st.session_state["hf_model"] = ""
if "collection_name" not in st.session_state:
    st.session_state["collection_name"] = ""
if "db_folder" not in st.session_state:
    st.session_state["db_folder"] = ""
if "use_existing_db" not in st.session_state:
    st.session_state["use_existing_db"] = True

# Function to list existing ChromaDB collections
def list_chromadb_collections(folder):
    client = chromadb.Client(Settings(persist_directory=folder))
    return client.list_collections()

# Function to create new collection in ChromaDB
def create_chromadb_collection(folder, collection_name):
    client = chromadb.Client(Settings(persist_directory=folder))
    return client.create_collection(name=collection_name)

# Hugging Face Authentication Screen
st.title("RAG Agent Setup")

# Input Hugging Face token and model
with st.form("hf_auth_form"):
    st.subheader("Enter Hugging Face Credentials")
    hf_token = st.text_input("Hugging Face Token", type="password")
    hf_model = st.text_input("Model Name (e.g., 'gpt-neo', 'distilbert')")
    submitted = st.form_submit_button("Save and Proceed")

    if submitted:
        st.session_state["hf_token"] = hf_token
        st.session_state["hf_model"] = hf_model
        st.success("Credentials saved! Proceed to database setup.")

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
        folder = st.text_input("Enter folder path for existing VectorDB collections")
        if folder:
            try:
                collections = list_chromadb_collections(folder)
                collection = st.selectbox("Choose a collection", collections)
                st.session_state["collection_name"] = collection
                st.session_state["db_folder"] = folder
                st.success(f"Selected Collection: {collection}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        folder = st.text_input("Enter folder path to save the new VectorDB")
        new_collection_name = st.text_input("Enter new collection name")
        uploaded_files = st.file_uploader("Upload files for VectorDB creation", accept_multiple_files=True)
        
        if folder and new_collection_name:
            st.session_state["collection_name"] = new_collection_name
            st.session_state["db_folder"] = folder
            if st.button("Create Collection"):
                try:
                    create_chromadb_collection(folder, new_collection_name)
                    st.success(f"New collection '{new_collection_name}' created at {folder}!")
                except Exception as e:
                    st.error(f"Error: {e}")

# Chat Interface
if st.session_state["collection_name"]:
    st.subheader("Chat Interface")

    # Chat input and display
    with st.container():
        st.text_input("Enter your query", key="user_query")
        if st.button("Get Response"):
            query = st.session_state.get("user_query", "")
            if query:
                # Here, integrate your RAG agent logic
                response = f"Response to '{query}' from model '{st.session_state['hf_model']}'"
                st.markdown(f"**User:** {query}")
                st.markdown(f"**Model:** {response}")
            else:
                st.warning("Please enter a query.")
