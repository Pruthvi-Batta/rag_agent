---

# RAG Agent: Retrieval-Augmented Generation

## Overview

The **RAG Agent** is a powerful solution that combines retrieval, augmentation, and generation techniques to provide natural language responses based on the context retrieved from a large corpus of text data. This approach enhances user queries with relevant context, enabling the model to generate accurate and contextually enriched responses—all while ensuring data privacy and avoiding costly re-training of models.

---

## Features

### Text Embedding
To enable similarity search and context retrieval:
- All texts are converted into **vector embeddings**.
- **Tokenization Levels**:
  1. **Sentence Level**: Split text into individual sentences.
  2. **Paragraph Level**: Split text into paragraphs.
  3. **Page Level**: Split text by page.
  4. **Max-Words Level**: Split text into custom tokens with a maximum word limit.

The chosen level determines how the text is split and converted into vectors.  
**Default Model**: ChromaDB's in-built embedding model is used for vectorization.

Once embeddings are created:
- They can be **saved** and **reused**, making this process a one-time effort for static datasets.

---

### RAG Workflow
1. **Retrieval**:
   - The text embeddings are searched for similarity with the given user prompt.
   - The **top N similar contexts** are retrieved based on vector distances.

2. **Augmentation**:
   - The retrieved contexts are prepended or appended to the user's query.
   - This enriched prompt is sent to the model for inference.

3. **Generation**:
   - The model processes the augmented prompt and generates a natural language response.
   - No fine-tuning of the model is required, ensuring:
     - Data privacy (offline usage of models).
     - Faster deployment and reduced compute costs.

**Key Benefit**: The solution enables a generative model without re-training, preserving data security while leveraging state-of-the-art capabilities.

---

## Pre-Requisites

1. **Hugging Face Token**:  
   - Obtain a token by creating an account on [Hugging Face](https://www.aimluniverse.in/use-hugging-face/#access-token) and generate an access token.
2. **Model Name**:  
   - Use any model compatible with Hugging Face's Transformers library. Select based on your application and available compute resources.
3. **System Requirements**:  
   - Ensure sufficient computational power based on the chosen model:
     - For smaller models, a CPU will suffice.
     - For larger models, a GPU is recommended for faster inference. Visit [PyTorch](https://pytorch.org) to install the correct GPU-compatible PyTorch version.

---

## Environment Setup

### Instructions
1. The environment setup is managed using **Miniconda**. 
2. Modify the following in the `Makefile` as needed:
   - `CONDA_BIN`: Path to your Conda binary (if Conda is already installed).
   - `ENV_NAME`: Desired environment name.

### Commands

#### Install Miniconda, Create Environment, and Install Dependencies
```bash
make
```

#### Install Miniconda Only
```bash
make install-miniconda
```

#### Create Environment from `env.yaml`
```bash
make create-env
```

#### Display Activation Instructions
```bash
make activate-env
```

**Note**:  
The default environment installs the **CPU version** of PyTorch. If you have a GPU, install the appropriate PyTorch version from [here](https://pytorch.org).

---

## Configuration

Set the necessary configuration parameters in the file:  
`config/rag_config.yaml`

Example:
```yaml
hugging_face:
  # personal access token from hugging face
  token: 

  # model name from hugging face
  model: Qwen/Qwen2.5-Coder-0.5B-Instruct

  # model type, that defines the pipeline based on the model
  task_type: "text-generation"
  
  # -1 for CPU, 0 or higher for GPU
  device: -1

chromadb:
  # available options: ['sentence', 'paragraph', 'page', 'max_words'] 
  tokenise_mode: 'paragraph'
  max_words: 100

rag_constraints:
  number_of_top_contexts: 5
```

---

## Usage

1. **Update Configuration**:  
   - Modify `config/rag_config.yaml` as per your requirements.

2. **Run the Application**:  
   Use the following command:
   ```bash
   make run
   ```

---

## Solution Highlights

- **One-Time Embedding Creation**:
  - Static datasets can have embeddings generated and stored for repeated use, minimizing processing time.

- **Contextual Augmentation**:
  - Dynamically retrieve and append the most relevant context to user queries for better model responses.

- **Data Privacy**:
  - Models are used offline, ensuring sensitive data is not shared with external APIs.

- **Versatile Tokenization**:
  - Flexibly tokenize data at sentence, paragraph, page, or custom word limits for tailored vectorization.

- **No Fine-Tuning**:
  - Avoid the cost and complexity of retraining models. The solution works seamlessly with pre-trained Hugging Face models.

- **Customizability**:
  - Configure tokenization levels, vector database paths, and model settings as per your needs.

---

## Acknowledgements

This solution leverages:
- [Hugging Face Transformers](https://huggingface.co/transformers): For state-of-the-art NLP models.
- [ChromaDB](https://www.trychroma.com/): For efficient vector storage and similarity search.
- [PyTorch](https://pytorch.org/): For model inference.

For questions or contributions, feel free to reach out or submit an issue on our repository.

---

This enhanced README provides clear formatting, detailed explanations, and user-friendly instructions to make your solution accessible to others!