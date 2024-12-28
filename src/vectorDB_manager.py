import os
import chromadb
from utils import config, logger

class ChromaDBHandler:
    def __init__(self, persist_path):
        """
        Initialize the ChromaDBHandler with a persistent directory path.
        :param persist_path: Path for the persistent database directory.
        """
        self.persist_path = persist_path
        self.client = chromadb.PersistentClient(
            path=self.persist_path
        )
        self.collection = None

    def create_or_replace_collection(self, collection_name):
        """
        Create or replace an existing collection with new data.
        :param collection_name: Name of the collection to create or replace.
        """

        # Drop existing collection if it exists
        if collection_name in [i.name for i in self.client.list_collections()]:
            self.client.delete_collection(collection_name)

        # Create the new collection
        self.collection = self.client.create_collection(name=collection_name)

        logger.info(f"Collection '{collection_name}' created and persisted at '{self.persist_path}'.")

    
    def add_tokens(self, text_tokens, metadata):
        logger.info("Adding Tokens to collection")
        if not isinstance(text_tokens, list) or not isinstance(metadata, list):
            raise ValueError("text_tokens and metadata must be lists.")
        if len(text_tokens) != len(metadata):
            raise ValueError("text_tokens and metadata must have the same length.")
        # Add data to the collection
        self.collection.add(
            ids=[f"id_{i}" for i in range(len(text_tokens))],
            documents=text_tokens,
            metadatas=metadata,
        )
    
    # Function to list existing ChromaDB collections
    def list_chromadb_collections(self):
        return self.client.list_collections()

    def load_existing_collection(self, collection_name):
        """
        Load an existing collection from the database.
        :param collection_name: Name of the collection to load.
        """
        self.collection = self.client.get_collection(name=collection_name)
        logger.info(f"Collection '{collection_name}' loaded.")

    def retrieve_top_n(self, query_text, n=config["rag_constraints"]["number_of_top_contexts"]):
        """
        Retrieve the top-N closest matches for a given query.
        :param query_text: The query text to search for.
        :param n: Number of top results to retrieve.
        :return: List of dictionaries containing matched documents and metadata.
        """
        if not self.collection:
            raise ValueError("No collection loaded. Please load or create a collection first.")

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n,
        )
        return results
