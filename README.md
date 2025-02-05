# Adaptive Retrieval-Augmented Generation Tool

This tool combines **Retrieval-Augmented Generation (RAG)** with **Agentic** capabilities to process and respond to user queries. It handles a variety of documents, allows for advanced document retrieval and hybrid search (BM25 + Vector Search), and utilizes state-of-the-art models from OpenAI (GPT-4).

The tool is built with **Langchain** for document retrieval, **Langgraph** for state-based workflows, and **Streamlit** for the user interface. It aims to deliver accurate, dynamic, and adaptive answers based on a wide variety of document types (PDF, DOCX, TXT, URLs).

## Features

- **Document Upload**: Upload PDFs, DOCX, and TXT files.
- **Hybrid Search**: Combine **vector search** and **BM25** for more accurate document retrieval.
- **Hallucination Detection**: Ensure the response is fully supported by the retrieved documents.
- **Adaptive Workflow**: Uses **Langgraph** to manage states and transitions in the query-answering process.
- **Agent-based Decision Making**: The system decides whether to retrieve additional data or proceed to answer generation.
- **Retrieval Tools**: Uses pre-configured tools and models to retrieve relevant information and generate answers.

## Requirements

To run the project, you need the following dependencies:

- `langchain`
- `langgraph`
- `streamlit`
- `openai`
- `pydantic`
- `sentence-transformers`
- `rank_bm25`
- `chromadb`
- `requests`

You can install the required dependencies using pip:

```bash
pip install -r requirements.txt

**Setup Instructions**
API Keys: Make sure to set the environment variables for your API keys.

1. OpenAI API Key: Set OPENAI_API_KEY environment variable.
Tavily API Key (optional for advanced use cases): Set TAVILY_API_KEY environment variable.
You can do this in your terminal before running the app:

export OPENAI_API_KEY="your_openai_api_key"
export TAVILY_API_KEY="your_tavily_api_key"


2. Download the Files: Make sure to have a folder data/ where you can upload documents for processing.

**Usage**

1. Run the App
Start the Streamlit app by running the following command:

streamlit run app.py

2. Upload Documents
You can upload your PDF, DOCX, or TXT files to be processed. The tool will extract the relevant information from these files.

3. Enter a Query
Once documents are uploaded, type your query in the input box. The system will process the query and respond with the most relevant answer, taking into account the content of the uploaded documents.

4. Review the Response
The system will show you a response generated using the RAG mechanism. The response will be checked for hallucinations (incorrect information that is not supported by the retrieved documents).

**How It Works**
Document Loading: Uploads and loads documents from various sources (URLs, file uploads).
Document Splitting: Documents are split into chunks for easier retrieval.
Embedding: The documents are embedded using OpenAI's embeddings.
Hybrid Search: Combines vector search and BM25 search for optimal document retrieval.
Hallucination Detection: Verifies whether the generated response is backed by the retrieved documents.
Adaptive RAG: The state machine (via Langgraph) determines if additional retrieval is needed or if the answer can be generated immediately.

**Code Structure**
app.py: Main entry point for the Streamlit app.
adaptive_rag_tool.py: Core logic of the Adaptive RAG tool.
utils.py: Utility functions for document loading, splitting, and retrieval.
requirements.txt: List of dependencies for the project.
License
This project is licensed under the MIT License.

