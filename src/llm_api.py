import pdb
from transformers import pipeline
from utils import logger, config
import os


class LLMHandler:
    def __init__(
        self, model_name: str, task: str = "text-generation", device: int = -1
    ):
        """
        Initialize the LLM handler with the specified model.

        :param model_name: Name of the Hugging Face model (e.g., "gpt2", "EleutherAI/gpt-neo-2.7B").
        :param task: Task to use the pipeline for (default is "text-generation").
        :param device: Device to use (-1 for CPU, 0 or higher for GPU).
        """
        self.model_name = model_name
        self.task = task
        self.device = device
        try:
            self.llm_pipeline = pipeline(task, model=model_name, device=device)
            print(
                f"Initialized model '{model_name}' for task '{task}' on device {device}."
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing model: {e}")

    def query_model(self, query: str, **kwargs) -> str:
        """
        Query the language model and return its response.

        :param query: The input query string.
        :param kwargs: Additional arguments for the pipeline (e.g., max_length, temperature).
        :return: Model's response as a string.
        """
        try:
            logger.info("Querying model")
            response = self.llm_pipeline(query, max_new_tokens=2048, temperature=0.1, **kwargs)
            # For tasks like text-generation, the response is a list of dictionaries
            if isinstance(response, list) and "generated_text" in response[0]:
                return [i['content'] for i in response[0]["generated_text"] if i['role'] == 'assistant'][0]
            return 'No response'
        except Exception as e:
            return f"Error querying model: {e}"

    def enhance_query(self, query: str, retrieved_context: str) -> str:
        """
        Enhance the query by combining it with a user-provided prompt.

        :param query: The original query string.
        :param prompt: The enhancement prompt.
        :return: Enhanced query string.
        """
        logger.info(f"Enhancing prompt: {query}")
        formatted_context = ""
        for meta, doc in zip(retrieved_context['metadatas'][0], retrieved_context['documents'][0]):
            formatted_context += f"Retrieved from document: {os.path.join(meta['file_path'], meta['file_name'])}/n"
            formatted_context += doc
            formatted_context += "/n/n"

        enhanced_query = [
            {
                "role": "system",
                "content": """
                    You are an AI assistant, helping me in the specific domain in RAG approach.
                    So when user prompt is given to you answer the query only on the provided context.
                    If the query cannot be answered using the context, then reply relevant context not retrieved, don't answer out of context.
                    Give concise and reponse as asked by the user. Default to short answers, give lengthy answers only when requested.
                """,
            },
            {
                "role": "user",
                "content": f"""
                    context: {formatted_context}
                    
                    User's Prompt: {query}
                """,
            },
        ]
        return enhanced_query
