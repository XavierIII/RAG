import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import langgraph
from langgraph.graph import StateGraph, END, START
from langchain.schema import Document
from langchain.tools.retriever import create_retriever_tool
from pydantic import BaseModel, Field
from typing import Annotated, Literal, List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain.prebuilt import tools_condition
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import ToolNode
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Set API keys
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["TAVILY_API_KEY"] = "your_tavily_api_key"

# URLs to index
doc_urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load and split documents
def load_documents(file_paths):
    """Load documents from various file formats."""
    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyMuPDFLoader(path)
        elif path.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        elif path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append(Document(page_content=text))
            continue
        else:
            continue
        docs.extend(loader.load())
    return docs

docs = []
for url in doc_urls:
    loader = WebBaseLoader(url)
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs)

# Initialize vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(doc_splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

# Define hybrid retrieval (BM25 + Vector Search)
def bm25_search(query, corpus):
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25.get_top_n(query.split(), corpus, n=5)

def hybrid_retrieval(query, vectorstore, text_corpus):
    vector_results = vectorstore.similarity_search(query, k=5)
    keyword_results = bm25_search(query, text_corpus)
    return list(set(vector_results + keyword_results))

# Define hallucination detection
def detect_hallucinations(documents, response):
    """Check if the response is supported by retrieved documents."""
    prompt = f"""
    Compare this response: {response}
    To these documents: {documents}
    Is the response fully supported? Answer 'yes' or 'no'.
    """
    return llm.invoke(prompt).content

# Define LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

# Define query rewriter
def rewrite_query(state):
    messages = state["messages"]
    question = messages[0].content
    msg = HumanMessage(content=f"Rewrite this question to improve retrieval: {question}")
    response = llm.invoke(msg)
    return {"messages": [response]}

# Define response generation
def generate(state):
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

# Define document grading function
def grade_documents(state):
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    prompt = f"Assess relevance: {docs} to question: {question}. Answer 'yes' or 'no'."
    score = llm.invoke(prompt).content
    return "generate" if score == "yes" else "rewrite"

# Define Agent State
class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", generate)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()

# Streamlit UI
st.title("Adaptive Retrieval-Augmented Generation Tool")
user_query = st.text_input("Enter your query:")
uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)
if uploaded_files:
    file_paths = [f"data/{file.name}" for file in uploaded_files]
    with open(file_paths, "wb") as f:
        f.write(uploaded_files.read())
    st.write("Files uploaded successfully!")

if st.button("Submit"):
    if user_query:
        response = graph.invoke({"messages": [("user", user_query)]})
        st.write("Response:", response)
    else:
        st.write("Please enter a query.")
